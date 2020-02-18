from collections import OrderedDict
from functools import partial
import glob
import copy
import sys
import os
if sys.version_info[0] < 3:
    import Tkinter as tkinter
    import tkFileDialog as filedialog
else:
    import tkinter
    from tkinter import filedialog
import warnings
from tkinter import ttk

import numpy as np
import matplotlib
mplv = matplotlib.__version__
mplvmaj = int(mplv.split('.')[0])
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
if mplvmaj > 2:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as mplnavtb
else:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as mplnavtb

from .. import definitions as xrsdefs 
from . import plot_xrsd_fit, draw_xrsd_fit
from .. import system as xrsdsys
from ..tools import ymltools as xrsdyml
from ..tools import profiler
from ..models import predict as xrsdpred
from ..models.train import train_from_dataframe
from ..models import load_models

q_default = np.linspace(0.,1.,100)
I_default = np.zeros(q_default.shape)

def run_gui():
    gui = XRSDFitGUI()
    gui.start()

# TODO: mouse-over pop-up help windows

# TODO: mousewheel should scroll whatever window contains the mouse pointer 

# TODO (low): when a combobox selection is rejected (raises an Exception),
#   get the associated combobox re-painted-
#   currently the value does get reset, 
#   but the cb does not repaint until it is focused-on...
#   try update_idletasks()?

# TODO (low): in _validate_param(), 
#   if param_key == 'constraint_expr', 
#   validate the expression with lmfit/asteval

# TODO (low): when a param is fixed or has a constraint set,
#   make the value entry widget read-only

# TODO (low): find a way to fix the errors 
#   that sometimes occur when the gui is closed
#   (_tkinter.TclError: invalid command name)
#   NOTE: on Windows, the error message is more informative:
#   this has something to do with the scrollbars

class XRSDFitGUI(object):

    def __init__(self):
        super(XRSDFitGUI, self).__init__()
        # start with a default system definition, q, and I(q)
        self.sys = xrsdsys.System()
        self.q = q_default
        self.I = I_default
        self.dI = None
        self.fit_gui = tkinter.Tk()
        self.fit_gui.protocol('WM_DELETE_WINDOW',self._cleanup)
        # setup the main gui objects
        self._build_gui()
        # create the widgets for control 
        self._build_control_widgets()
        # create the plots
        self._build_plot_widgets()
        self.fit_gui.geometry('1100x750')
        self._draw_plots()
        self.data_file_map = OrderedDict() 

    def start(self):
        # start the tk loop
        self.fit_gui.mainloop()

    def _cleanup(self):
        # remove references to all gui objects, widgets, etc. 
        self.fit_gui.quit()
        self.fit_gui.destroy()

    def _build_gui(self):
        self.fit_gui.title('xrsd pattern analyzer')
        # a horizontal scrollbar and a main canvas belong to the main gui:
        scrollbar = tkinter.Scrollbar(self.fit_gui,orient='horizontal')
        self.main_canvas = tkinter.Canvas(self.fit_gui)
        scrollbar.pack(side=tkinter.BOTTOM,fill=tkinter.X)
        self.main_canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)
        scrollbar.config(command=self.main_canvas.xview)
        self.main_canvas.config(xscrollcommand=scrollbar.set)
        # the main widget will be a frame,
        # displayed as a window item on the main canvas:
        self.main_frame = tkinter.Frame(self.main_canvas,bd=4,relief=tkinter.SUNKEN)
        self.main_frame.grid_columnconfigure(0,weight=1)
        self.main_frame.grid_columnconfigure(1,weight=0,minsize=400)
        self.main_frame.grid_rowconfigure(0,weight=1)
        main_frame_window = self.main_canvas.create_window(0,0,window=self.main_frame,anchor='nw')
        # _canvas_configure() ensures that the window item and scrollbar
        # remain the correct size for the underlying widget
        main_canvas_configure = partial(self._canvas_configure,self.main_canvas,self.main_frame,main_frame_window)  
        self.main_canvas.bind('<Configure>',main_canvas_configure)

    @staticmethod
    def _canvas_configure(canvas,widget,window,event=None):
        # Resize the frame to match the canvas.
        # The window is the "canvas item" that displays the widget.
        minw = widget.winfo_reqwidth()
        minh = widget.winfo_reqheight()
        if canvas.winfo_width() > minw:
            minw = canvas.winfo_width()
        if canvas.winfo_height() > minh:
            minh = canvas.winfo_height()
        canvas.itemconfigure(window,width=minw,height=minh)
        canvas.config(scrollregion=canvas.bbox(tkinter.ALL))

    def _build_plot_widgets(self):
        # the main frame contains a frame on the left,
        # containing a canvas, which contains a window item,
        # which displays a view on a plot widget 
        # built from FigureCanvasTkAgg.get_tk_widget()
        plot_frame = tkinter.Frame(self.main_frame,bd=4,relief=tkinter.SUNKEN)
        plot_frame.grid(row=0,column=0,sticky='nesw',padx=2,pady=2)
        self.fig,I_comp = plot_xrsd_fit(sys=self.sys,show_plot=False)
        plot_frame_canvas = tkinter.Canvas(plot_frame)
        yscr = tkinter.Scrollbar(plot_frame)
        yscr.pack(side=tkinter.RIGHT,fill='y')
        plot_frame_canvas.pack(fill='both',expand=True)
        plot_frame_canvas.config(yscrollcommand=yscr.set)
        yscr.config(command=plot_frame_canvas.yview)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig,plot_frame_canvas)
        self.plot_canvas = self.mpl_canvas.get_tk_widget()
        plot_toolbar = mplnavtb(self.mpl_canvas,plot_frame)
        plot_toolbar.update()
        plot_canvas_window = plot_frame_canvas.create_window(0,0,window=self.plot_canvas,anchor='nw')
        self.plot_canvas_configure = partial(self._canvas_configure,
            plot_frame_canvas,self.plot_canvas,plot_canvas_window)
        plot_frame_canvas.bind("<Configure>",self.plot_canvas_configure)
        self.mpl_canvas.draw()

    def _build_control_widgets(self):
        # the main frame contains a frame on the right,
        # containing a canvas, which contains a window item,
        # which displays a view on a frame full of entry widgets and labels, 
        # which are used to control parameters, settings, etc. 
        control_frame = tkinter.Frame(self.main_frame)
        control_frame.grid(row=0,column=1,sticky='nesw',padx=2,pady=2)
        control_frame_canvas = tkinter.Canvas(control_frame)
        #control_frame.bind_all("<MouseWheel>", partial(self.on_mousewheel,control_frame_canvas))
        #control_frame.bind_all("<Button-4>", partial(self.on_trackpad,control_frame_canvas))
        #control_frame.bind_all("<Button-5>", partial(self.on_trackpad,control_frame_canvas))
        yscr = tkinter.Scrollbar(control_frame)
        yscr.pack(side=tkinter.RIGHT,fill='y')
        control_frame_canvas.pack(fill='both',expand=True)
        control_frame_canvas.config(yscrollcommand=yscr.set)
        yscr.config(command=control_frame_canvas.yview)
        self.control_widget = tkinter.Frame(control_frame_canvas)
        self.control_widget.grid_columnconfigure(0,weight=1)
        control_canvas_window = control_frame_canvas.create_window((0,0),window=self.control_widget,anchor='nw')
        self.control_canvas_configure = partial(self._canvas_configure,
            control_frame_canvas,self.control_widget,control_canvas_window)
        control_frame_canvas.bind("<Configure>",self.control_canvas_configure)
        # set empty data structures to keep references to widgets and variables    
        self._reset_control_widgets()
        # create widgets and variables
        self._create_control_widgets()

    def _reset_control_widgets(self):
        # reset data structures for maintaining refs to widgets and vars
        self._frames = OrderedDict(
            noise_model=None,
            populations=OrderedDict(),
            parameters=OrderedDict(),
            settings=OrderedDict(),
            new_population=None,
            fit_control=None,
            io_control=None
            )
        self._vars = OrderedDict(
            noise_model=None,
            structures=OrderedDict(),
            form_factors=OrderedDict(),
            parameters=OrderedDict(),
            settings=OrderedDict(),
            new_population_name=None,
            fit_control=OrderedDict(),
            io_control=OrderedDict()
            )
        self._widgets = OrderedDict(
            data_file_cb=None,
            output_file_display=None,
            param_loader_cb=None,
            io_control=OrderedDict()
            )

    def _create_control_widgets(self):
        self._create_io_control_frame()
        self._create_fit_control_frame()
        self._create_noise_frame()
        for pop_nm in self.sys.populations.keys():
            self._create_pop_frame(pop_nm)
        self._create_new_pop_frame()
        self._pack_population_frames()

    def _create_io_control_frame(self):
        iof = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        iof.grid_columnconfigure(0,weight=1)
        iof.grid_columnconfigure(1,weight=1)
        iof.grid_columnconfigure(2,weight=1)
        iof.grid_rowconfigure(2,minsize=10)
        iof.grid_rowconfigure(6,minsize=10)
        iof.grid_rowconfigure(10,minsize=10)
        iof.grid_rowconfigure(13,minsize=10)
        self._frames['io_control'] = iof
        dfvar = tkinter.StringVar(iof)
        dfvar.trace('w',self._update_data_file)
        pfvar = tkinter.StringVar(iof)
        pfvar.trace('w',self._load_params_from_yml)

        self._vars['io_control']['data_dir'] = tkinter.StringVar(iof)
        self._vars['io_control']['search_regex'] = tkinter.StringVar(iof)
        self._vars['io_control']['data_file'] = dfvar
        self._vars['io_control']['output_file'] = tkinter.StringVar(iof)
        self._vars['io_control']['params_file'] = pfvar
        self._vars['io_control']['training_set_dir'] = tkinter.StringVar(iof)
        self._vars['io_control']['model_output_dir'] = tkinter.StringVar(iof)
        self._vars['io_control']['models_dir'] = tkinter.StringVar(iof)
        self._vars['io_control']['search_mode'] = tkinter.StringVar(iof)
        self._vars['io_control']['create_new_files_flag'] = tkinter.BooleanVar(iof)
        self._vars['io_control']['experiment_id'] = tkinter.StringVar(iof)
        self._vars['io_control']['source_wavelength'] = tkinter.DoubleVar(iof)
        self._widgets['io_control']['training_set_list'] = tkinter.Listbox(iof)

        dfl = tkinter.Label(iof,text='data directory:',anchor='e')
        dfe = tkinter.Entry(iof,state='readonly',textvariable=self._vars['io_control']['data_dir'],justify='right')
        dfbb = tkinter.Button(iof,text='Browse...',width=8,command=self._browse_data_files)
        dfl.grid(row=0,column=0,sticky='w')
        dfe.grid(row=1,column=0,columnspan=3,sticky='ew')
        dfbb.grid(row=0,column=1,columnspan=2,sticky='ew')

        # this creates and packs the data file selection menu:
        self._set_data_files()

        datfl = tkinter.Label(iof,text='data file:',anchor='e')
        prevb = tkinter.Button(iof,text='Previous',width=8,command=self._previous_data_file)
        nxtb = tkinter.Button(iof,text='Next',width=8,command=self._next_data_file)
        datfl.grid(row=3,column=0,sticky='w')
        prevb.grid(row=5,column=0,sticky='w')
        nxtb.grid(row=5,column=2,sticky='e')

        sysdefl = tkinter.Label(iof,text='output file:',anchor='e')
        sysdefl.grid(row=7,column=0,sticky='w')
        #sysfne = tkinter.Entry(iof,state='readonly',textvariable=self._vars['io_control']['output_file'],justify='right')
        sysfne = tkinter.Entry(iof,state='readonly',textvariable=self._vars['io_control']['output_file'])
        sysfne.grid(row=8,column=0,columnspan=3,sticky='ew')
        sysfsvb = tkinter.Button(iof,text='Save',width=8,command=self._save_output_file) 
        sysfldb = tkinter.Button(iof,text='Load',width=8,command=self._load_output_file) 
        sysfsvb.grid(row=9,column=0,sticky='w')
        sysfldb.grid(row=9,column=2,sticky='e')
        self._widgets['output_file_display'] = sysfne

        ymlfl = tkinter.Label(iof,text='load parameters from:',anchor='e')
        ymlfl.grid(row=11,column=0,sticky='w')

        modl = tkinter.Label(iof,text='train/load models:',anchor='e')
        modbb = tkinter.Button(iof,text='Browse...',width=8,command=self._browse_models)
        modl.grid(row=14,column=0,sticky='w')
        modbb.grid(row=14,column=1,columnspan=2,sticky='ew')

        iof.grid(row=0,pady=2,padx=2,sticky='ew')

    def _browse_models(self,*args):
        browser_popup = tkinter.Toplevel(master=self.fit_gui)
        browser_popup.geometry('450x600')
        browser_popup.title('modeling data browser')
        main_canvas = tkinter.Canvas(browser_popup)
        main_canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)
        main_frame = tkinter.Frame(main_canvas,bd=4,padx=10,pady=10)
        main_frame_window = main_canvas.create_window(0,0,window=main_frame,anchor='nw')
        main_canvas_configure = partial(self._canvas_configure,main_canvas,main_frame,main_frame_window)  
        main_canvas.bind("<Configure>",main_canvas_configure)
        entry_frame = tkinter.Frame(main_frame,bd=4,padx=10,pady=10,relief=tkinter.GROOVE)
        entry_frame.grid_columnconfigure(0,weight=2)
        entry_frame.grid_columnconfigure(1,weight=1)
        entry_frame.grid_columnconfigure(2,weight=1)
        entry_frame.grid_rowconfigure(2,minsize=20)
        entry_frame.grid_rowconfigure(5,minsize=20)
        # entry frame widgets
        odirl = tkinter.Label(entry_frame,text='Output trained models to:',anchor='w')
        ddirl = tkinter.Label(entry_frame,text='Train from dataset:',anchor='w')
        mdirl = tkinter.Label(entry_frame,text='Load trained models:',anchor='w')

        # get a safe default output directory for models
        default_output_dir = os.path.join(os.getcwd(),'xrsdkit_models')
        ii=1
        while os.path.exists(default_output_dir):
            default_output_dir = os.path.join(os.getcwd(),'xrsdkit_models_{}'.format(ii))
            ii+=1
        self._vars['io_control']['model_output_dir'].set(default_output_dir) 

        odirbb = tkinter.Button(entry_frame,text='Browse',command=partial(
            self._browse_for_directory,browser_popup,
            self._vars['io_control']['model_output_dir'],
            'Select output directory for modeling data'
            ))
        ddirbb = tkinter.Button(entry_frame,text='Browse',command=partial(
            self._browse_for_directory,browser_popup,
            self._vars['io_control']['training_set_dir'],
            'Select training dataset directory'
            ))
        mdirbb = tkinter.Button(entry_frame,text='Browse',command=partial(
            self._browse_for_directory,browser_popup,
            self._vars['io_control']['models_dir'],
            'Select directory of trained xrsdkit models'
            ))
        odirent = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['model_output_dir'])
        ddirent = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['training_set_dir'])
        mdirent = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['models_dir'])
        # display widgets
        display_frame = tkinter.Frame(main_frame,bd=4,padx=10,pady=10,relief=tkinter.GROOVE)
        display = tkinter.Listbox(display_frame)
        display.pack(fill=tkinter.BOTH,expand=True,padx=2,pady=2)
        # controls for launching training etc
        trainbtn = tkinter.Button(entry_frame,text='Train...',command=partial(self._train_models,display))
        loadbtn = tkinter.Button(entry_frame,text='Load',command=partial(self._load_models,display))
        # widget packing
        odirl.grid(row=0,column=0,sticky='ew')
        odirbb.grid(row=0,column=1,sticky='ew')
        odirent.grid(row=1,column=0,columnspan=3,sticky='ew')
        ddirl.grid(row=3,column=0,sticky='ew')
        ddirbb.grid(row=3,column=1,sticky='ew')
        trainbtn.grid(row=3,column=2,sticky='ew')
        ddirent.grid(row=4,column=0,columnspan=3,sticky='ew')
        mdirl.grid(row=6,column=0,sticky='ew')
        mdirbb.grid(row=6,column=1,sticky='ew')
        loadbtn.grid(row=6,column=2,sticky='ew')
        mdirent.grid(row=7,column=0,columnspan=3,sticky='ew')
        # frame packing 
        entry_frame.pack(side=tkinter.TOP,fill=tkinter.X,expand=False,padx=2,pady=2)
        display_frame.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=True,padx=2,pady=2)
        # wait for the browser to close before continuing main loop 
        self.fit_gui.wait_window(browser_popup)

    def _train_models(self,display):
        # TODO: input widget for downsampling distance?
        # TODO: toggles for hyperparam selection? feature selection?
        dataset_dir = self._vars['io_control']['training_set_dir'].get()
        output_dir = self._vars['io_control']['model_output_dir'].get() 
        model_config_path = os.path.join(output_dir,'model_config.yml')
        self._print_to_listbox(display,'LOADING DATASET FROM: {}'.format(dataset_dir))
        df, idx_df = xrsdyml.read_local_dataset([dataset_dir],downsampling_distance=1.,
                message_callback=partial(self._print_to_listbox,display))
        self._print_to_listbox(display,'---- FINISHED LOADING DATASET ----')
        self._print_to_listbox(display,'BEGINNING TO TRAIN MODELS')
        self._print_to_listbox(display,'MODEL CONFIG FILE PATH: {}'.format(model_config_path))
        reg_mods, cls_mods = train_from_dataframe(df, 
                train_hyperparameters=False, select_features=False,
                output_dir=output_dir, model_config_path=model_config_path,
                message_callback=partial(self._print_to_listbox,display)
                )
        self._print_to_listbox(display,'---- FINISHED TRAINING ----')

    def _load_models(self,display):
        models_dir = self._vars['io_control']['models_dir'].get()
        self._print_to_listbox(display,'LOADING MODELS FROM: {}'.format(models_dir)) 
        load_models(models_dir)
        self._print_to_listbox(display,'---- FINISHED LOADING MODELS ----'.format(models_dir)) 

    def _print_to_listbox(self,lb,msg):
        lb.insert(tkinter.END,msg)
        lb.see(tkinter.END)
        self.fit_gui.update_idletasks()

    def _browse_data_files(self,*args):
        #self._widgets['io_control']['experiment_id_label'] = None 
        #self._widgets['io_control']['experiment_id_entry'] = None 
        #self._widgets['io_control']['wavelength_label'] = None
        #self._widgets['io_control']['wavelength_entry'] = None
        #self._widgets['io_control']['new_files_button'] = None 

        browser_popup = tkinter.Toplevel(master=self.fit_gui)
        browser_popup.geometry('500x500')
        browser_popup.title('data browser')
        browser_popup.protocol('WM_DELETE_WINDOW',self._cleanup_data_browser)

        main_canvas = tkinter.Canvas(browser_popup)
        main_frame = tkinter.Frame(main_canvas,bd=4,padx=10,pady=10)
        main_frame_window = main_canvas.create_window(0,0,window=main_frame,anchor='nw')
        main_canvas_configure = partial(self._canvas_configure,main_canvas,main_frame,main_frame_window)  
        main_canvas.bind("<Configure>",main_canvas_configure)
        scrollbar = tkinter.Scrollbar(browser_popup,orient='vertical')
        scrollbar.pack(side=tkinter.RIGHT,fill=tkinter.Y)
        main_canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)
        scrollbar.config(command=main_canvas.yview)
        main_canvas.config(yscrollcommand=scrollbar.set)

        display_frame = tkinter.Frame(main_frame,bd=4,padx=10,pady=10,relief=tkinter.GROOVE)
        dfplist = tkinter.Listbox(display_frame)
        sfplist = tkinter.Listbox(display_frame)

        entry_frame = tkinter.Frame(main_frame,bd=4,padx=10,pady=10,relief=tkinter.GROOVE)
        entry_frame.grid_columnconfigure(0,weight=1)
        entry_frame.grid_columnconfigure(1,weight=1)
        entry_frame.grid_columnconfigure(2,weight=2)
        entry_frame.grid_rowconfigure(2,minsize=10)
        entry_frame.grid_rowconfigure(6,minsize=10)
        # widgets for setting data files directory
        ddirl = tkinter.Label(entry_frame,text='data directory:',anchor='w')
        drxl = tkinter.Label(entry_frame,text='filter:',anchor='w')
        ddirent = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['data_dir'])
        drxent = tkinter.Entry(entry_frame,width=6,textvariable=self._vars['io_control']['search_regex'])
        self._vars['io_control']['search_regex'].set('*.dat')
        ddirbb = tkinter.Button(entry_frame,text='Browse',command=partial(
            self._browse_for_directory,browser_popup,
            self._vars['io_control']['data_dir'],
            'Select scattering data directory'
            ))
        ddirl.grid(row=0,column=0,sticky='sew')
        ddirbb.grid(row=0,column=1,sticky='sew')
        ddirent.grid(row=1,column=0,columnspan=2,sticky='ew')
        drxl.grid(row=0,column=2,sticky='w')
        drxent.grid(row=1,column=2,sticky='ew')

        smodelbl = tkinter.Label(entry_frame,text='search for:',anchor='e')
        smodebtn0 = tkinter.Radiobutton(entry_frame,text='data files',
                    variable=self._vars['io_control']['search_mode'],value='data_files')
        smodebtn1 = tkinter.Radiobutton(entry_frame,text='output files',
                    variable=self._vars['io_control']['search_mode'],value='output_files')
        self._vars['io_control']['search_mode'].set('data_files')

        exptidl = tkinter.Label(entry_frame,text='experiment id:',anchor='w',state='disabled')
        srcwll = tkinter.Label(entry_frame,text='wavelength (Angstroms):',anchor='w',state='disabled')
        exptide = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['experiment_id'],width=8,state='disabled')
        srcwle = tkinter.Entry(entry_frame,textvariable=self._vars['io_control']['source_wavelength'],width=8,state='disabled')

        self._vars['io_control']['create_new_files_flag'].set(False)
        newfilesbtn = tkinter.Checkbutton(entry_frame,
                    text='Create new output files:',
                    variable=self._vars['io_control']['create_new_files_flag'],
                    anchor='w')

        exptidl.grid(row=4,column=1,sticky='w')
        exptide.grid(row=4,column=2,sticky='ew')
        srcwll.grid(row=5,column=1,sticky='w')
        srcwle.grid(row=5,column=2,sticky='ew')

        # need references to these for callbacks
        self._widgets['io_control']['experiment_id_label'] = exptidl
        self._widgets['io_control']['experiment_id_entry'] = exptide
        self._widgets['io_control']['wavelength_label'] = srcwll
        self._widgets['io_control']['wavelength_entry'] = srcwle
        self._widgets['io_control']['new_files_button'] = newfilesbtn 

        self._vars['io_control']['create_new_files_flag'].trace('w',self._toggle_new_file_entries)
        self._vars['io_control']['search_mode'].trace('w',self._update_search_mode)

        smodelbl.grid(row=3,column=0,sticky='w')
        smodebtn0.grid(row=4,column=0,sticky='w')
        smodebtn1.grid(row=5,column=0,sticky='w')

        newfilesbtn.grid(row=3,column=1,columnspan=2,sticky='w')

        sexprb = tkinter.Button(entry_frame,text='Execute Search',command=partial(self._execute_search,dfplist,sfplist))
        sexprb.grid(row=9,column=0,columnspan=3,sticky='nsew')
        finbtn = tkinter.Button(entry_frame,text='Finish',
            command=partial(self._get_data_files_from_browser,dfplist,sfplist,browser_popup))
        finbtn.grid(row=10,column=0,columnspan=3,sticky='nsew')

        display_frame.grid_columnconfigure(0,weight=1)
        display_frame.grid_columnconfigure(1,weight=1)
        display_frame.grid_rowconfigure(1,weight=1)
        # widgets for displaying file lists:
        dfl = tkinter.Label(display_frame,text='data files:',anchor='w')
        sfl = tkinter.Label(display_frame,text='output files:',anchor='w')
        dfl.grid(row=0,column=0,sticky='w')
        sfl.grid(row=0,column=1,sticky='w')

        dfl.config(justify=tkinter.RIGHT)
        sfl.config(justify=tkinter.RIGHT)

        dfplist.grid(row=1,column=0,sticky='nsew') 
        sfplist.grid(row=1,column=1,sticky='nsew')

        # finally, pack frames into the main widget
        entry_frame.pack(side=tkinter.TOP,fill=tkinter.X,expand=False,padx=2,pady=2)
        display_frame.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=True,padx=2,pady=2)

        # wait for the browser to close before continuing main loop 
        self.fit_gui.wait_window(browser_popup)

    def _cleanup_data_browser(self,browser):
        self._vars['io_control']['create_new_files_flag'].trace_vdelete()
        self._vars['io_control']['search_mode'].trace_vdelete()
        browser.destroy() 

    def _toggle_new_file_entries(self,*event_args):
        flag = self._vars['io_control']['create_new_files_flag'].get()

        for widg in [ self._widgets['io_control']['experiment_id_label'], \
                self._widgets['io_control']['experiment_id_entry'], \
                self._widgets['io_control']['wavelength_label'], \
                self._widgets['io_control']['wavelength_entry'] ]:
            try:
                if flag:
                    widg.config(state='normal')
                else:
                    widg.config(state='disabled')
            except:
                pass
        return True

    def _update_search_mode(self,*event_args):
        search_mode = self._vars['io_control']['search_mode'].get()

        if search_mode == 'data_files':
            self._widgets['io_control']['new_files_button'].config(state='normal')
        elif search_mode == 'output_files':
            self._vars['io_control']['create_new_files_flag'].set(False)
            #new_files_btn.deselect() 
            self._widgets['io_control']['new_files_button'].config(state='disabled')

    def _execute_search(self,data_file_listbox,output_file_listbox):
        data_dir = self._vars['io_control']['data_dir'].get()
        data_rx = self._vars['io_control']['search_regex'].get()
        files = []
        if data_dir and data_rx:
            data_expr = os.path.join(data_dir,data_rx)
            paths = glob.glob(data_expr)
            files = [os.path.split(p)[1] for p in paths]

        search_mode = self._vars['io_control']['search_mode'].get()
        output_files = OrderedDict() 
        if search_mode == 'data_files':
            new_files_flag = self._vars['io_control']['create_new_files_flag'].get()
            exptid = self._vars['io_control']['experiment_id'].get()
            src_wl = self._vars['io_control']['source_wavelength'].get()

            # build all output (yaml) file names,
            # collect any existing sample_ids 
            sample_ids = []
            for fn in files:
                ymlfn = os.path.splitext(fn)[0]+'.yml'
                output_files[fn] = ymlfn 
                outfile = os.path.join(data_dir,ymlfn)
                if os.path.exists(outfile):
                    sys = xrsdyml.load_sys_from_yaml(outfile)
                    sample_ids.append(sys.sample_metadata['sample_id'])

            # create new output files if called for
            sampl_idx = 0
            for fn in files:
                outfile = output_files[fn]
                if not os.path.exists(outfile):
                    if new_files_flag:
                        samplid = exptid+'_'+str(sampl_idx)
                        while samplid in sample_ids:
                            sampl_idx += 1
                            samplid = exptid+'_'+str(sampl_idx)
                        new_sys = xrsdsys.System()
                        new_sys.sample_metadata = {'experiment_id':exptid,\
                                'sample_id':samplid,'source_wavelength':src_wl,\
                                'data_file':fn}
                        xrsdyml.save_sys_to_yaml(outfile,new_sys)
                    #else:
                    #    output_files.pop(fn)
        elif search_mode == 'output_files':
            for fn in files:
                fp = os.path.join(data_dir,fn)
                try:
                    sys = xrsdyml.load_sys_from_yaml(fp)
                    datfn = sys.sample_metadata['data_file'] 
                    output_files[datfn] = fn
                except:
                    pass
        data_file_listbox.delete(0,tkinter.END)
        output_file_listbox.delete(0,tkinter.END)
        data_file_listbox.insert(0,*list(output_files.keys()))
        output_file_listbox.insert(0,*list(output_files.values()))

    def _get_data_files_from_browser(self,data_file_listbox,output_file_listbox,browser_popup):

        # TODO: review/test from here next

        df_list = data_file_listbox.get(0,tkinter.END)
        sf_list = output_file_listbox.get(0,tkinter.END)
        df_map = OrderedDict((df,sf) for df,sf in zip(df_list,sf_list))
        self._set_data_files(df_map)
        browser_popup.destroy()

    def _browse_for_directory(self,parent_widget,dir_entry_var,title=''):
        browser_root = os.getcwd()
        data_dir = filedialog.askdirectory(
            parent=parent_widget,
            initialdir=browser_root,
            title=title
            )
        dir_entry_var.set(data_dir)

    def _set_data_files(self,all_data_files={}):
        self.data_file_map = all_data_files

        dfcb = ttk.Combobox(self._frames['io_control'],
            textvariable=self._vars['io_control']['data_file'],
            values=list(all_data_files.keys()),
            state='readonly'
            )
        ymlfcb = ttk.Combobox(self._frames['io_control'],
            textvariable=self._vars['io_control']['params_file'],
            values=list(all_data_files.values()),
            state='readonly'
            ) 

        if self._widgets['data_file_cb']: 
            self._widgets['data_file_cb'].grid_forget()
        if self._widgets['param_loader_cb']: 
            self._widgets['param_loader_cb'].grid_forget()
        dfcb.grid(row=4,column=0,columnspan=3,sticky='ew')
        ymlfcb.grid(row=12,column=0,columnspan=3,sticky='ew')
        self._widgets['data_file_cb'] = dfcb
        self._widgets['param_loader_cb'] = ymlfcb 
        if self.data_file_map:
            self._next_data_file()

    # TODO: something more elegant for previous/next data file selection

    def _next_data_file(self,*args):
        current_file = self._vars['io_control']['data_file'].get()
        file_list = list(self.data_file_map.keys())
        nfiles = len(file_list)
        if current_file in file_list:
            current_file_idx = file_list.index(current_file)
            next_file_idx = min([nfiles-1,current_file_idx+1])
        else:
            next_file_idx = 0
        if next_file_idx < nfiles: 
            next_file = file_list[next_file_idx]
            if not current_file == next_file:
                # setting the var triggers self._update_data_file()
                self._vars['io_control']['data_file'].set(next_file)

    def _previous_data_file(self,*args):
        current_file = self._vars['io_control']['data_file'].get()
        file_list = list(self.data_file_map.keys())
        prev_file = ''
        if current_file:
            current_file_idx = file_list.index(current_file)
            prev_file_idx = current_file_idx-1
            if prev_file_idx>=0: prev_file = file_list[prev_file_idx]
        if not current_file == prev_file:
            # setting the var triggers self._update_data_file()
            self._vars['io_control']['data_file'].set(prev_file)

    def _save_output_file(self,*args):
        # TODO: if file already exists, warn user about overwrite
        data_dir = self._vars['io_control']['data_dir'].get()
        outfile = self._vars['io_control']['output_file'].get()
        if outfile and os.path.exists(data_dir):
            outfile_path = os.path.join(data_dir,outfile)
            xrsdyml.save_sys_to_yaml(outfile_path,self.sys)

    def _load_output_file(self,*args):
        data_dir = self._vars['io_control']['data_dir'].get()
        outfile = self._vars['io_control']['output_file'].get()
        outfile_path = os.path.join(data_dir,outfile)
        if os.path.exists(outfile_path):
            new_sys = xrsdyml.load_sys_from_yaml(outfile_path)
        else:
            new_sys = xrsdsys.System()
        self._set_system(new_sys)

    def _update_data_file(self,*event_args):
        data_dir = self._vars['io_control']['data_dir'].get()
        df = self._vars['io_control']['data_file'].get()
        if df:
            dpath = os.path.join(data_dir,df)
            q_I = np.loadtxt(dpath)
            self.q = q_I[:,0]
            self.I = q_I[:,1]
            self.dI = None
            if q_I.shape[1] > 2:
                self.dI = q_I[:,2]
            sysf = self.data_file_map[df]
            if not sysf:
                sysf = os.path.splitext(df)[0]+'.yml'
            self._vars['io_control']['output_file'].set(sysf)
            self._vars['io_control']['params_file'].set(sysf)
            self._widgets['output_file_display'].xview(len(sysf))
            self._load_output_file()
        else:
            self.q = q_default
            self.I = I_default
            self.dI = None
            self._vars['io_control']['output_file'].set('')
            self._set_system(xrsdsys.System())

    def _load_params_from_yml(self,*event_args):
        ymlf = self._vars['io_control']['params_file'].get()
        output_ymlf = self._vars['io_control']['output_file'].get()
        data_dir = self._vars['io_control']['data_dir'].get()
        ymlpath = os.path.join(data_dir,ymlf)
        output_ymlpath = os.path.join(data_dir,output_ymlf)
        if os.path.exists(ymlpath) and not ymlf == output_ymlpath:
            new_sys = self.sys.clone() 
            for pop_nm in list(new_sys.populations.keys()):
                new_sys.remove_population(pop_nm)
            params_sys = xrsdyml.load_sys_from_yaml(ymlpath)
            new_sys.update_from_dict(params_sys.populations)
            new_sys.update_noise_model(params_sys.noise_model.to_dict())
            self._set_system(new_sys)

    def _create_fit_control_frame(self):
        cf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        cf.grid_columnconfigure(0,weight=0)
        cf.grid_columnconfigure(1,weight=1)
        cf.grid_columnconfigure(2,weight=1)
        self._frames['fit_control'] = cf

        self._vars['fit_control']['experiment_id'] = tkinter.StringVar(cf)
        self._vars['fit_control']['experiment_id'].set(self.sys.sample_metadata['experiment_id'])
        self._vars['fit_control']['sample_id'] = tkinter.StringVar(cf)
        self._vars['fit_control']['sample_id'].set(self.sys.sample_metadata['sample_id'])
        self._vars['fit_control']['wavelength'] = tkinter.DoubleVar(cf)
        self._vars['fit_control']['wavelength'].set(self.sys.sample_metadata['source_wavelength'])
        self._vars['fit_control']['objective'] = tkinter.StringVar(cf)
        self._vars['fit_control']['error_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['logI_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['error_weighted'].set(self.sys.fit_report['error_weighted'])
        self._vars['fit_control']['logI_weighted'].set(self.sys.fit_report['logI_weighted'])
        self._vars['fit_control']['q_range'] = [tkinter.DoubleVar(cf),tkinter.DoubleVar(cf)]
        self._vars['fit_control']['q_range'][0].set(self.sys.fit_report['q_range'][0])
        self._vars['fit_control']['q_range'][1].set(self.sys.fit_report['q_range'][1])
        self._vars['fit_control']['good_fit'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['good_fit'].set(self.sys.fit_report['good_fit'])

        exptidl = tkinter.Label(cf,text='experiment id:',anchor='e')
        exptide = self.connected_entry(cf,self._vars['fit_control']['experiment_id'],self._set_experiment_id,10)
        sampidl = tkinter.Label(cf,text='sample id:',anchor='e')
        sampide = self.connected_entry(cf,self._vars['fit_control']['sample_id'],self._set_sample_id,10)
        exptidl.grid(row=5,column=0,sticky='e')
        exptide.grid(row=5,column=1,columnspan=2,sticky='ew')
        sampidl.grid(row=6,column=0,sticky='e')
        sampide.grid(row=6,column=1,columnspan=2,sticky='ew')

        wll = tkinter.Label(cf,text='wavelength:',anchor='e')
        wle = self.connected_entry(cf,self._vars['fit_control']['wavelength'],self._set_wavelength,10)
        wll.grid(row=7,column=0,sticky='e')
        wle.grid(row=7,column=1,columnspan=2,sticky='ew')

        q_range_lbl = tkinter.Label(cf,text='q-range:',anchor='e')
        q_range_lbl.grid(row=8,column=0,sticky='e')
        q_lo_ent = self.connected_entry(cf,self._vars['fit_control']['q_range'][0],partial(self._set_q_range,0),8) 
        q_hi_ent = self.connected_entry(cf,self._vars['fit_control']['q_range'][1],partial(self._set_q_range,1),8) 
        q_lo_ent.grid(row=8,column=1,sticky='ew')
        q_hi_ent.grid(row=8,column=2,sticky='ew')

        ewtcb = self.connected_checkbutton(cf,self._vars['fit_control']['error_weighted'],self._set_error_weighted,'error weighted')
        ewtcb.grid(row=9,column=0,sticky='e')
        logwtcb = self.connected_checkbutton(cf,self._vars['fit_control']['logI_weighted'],self._set_logI_weighted,'log(I) weighted')
        logwtcb.grid(row=10,column=0,sticky='e')

        fitbtn = tkinter.Button(cf,text='Fit',width=8,command=self._fit)
        fitbtn.grid(row=9,column=1,rowspan=2,sticky='nesw')
        estbtn = tkinter.Button(cf,text='Estimate',width=8,command=self._estimate)
        estbtn.grid(row=9,column=2,rowspan=2,sticky='nesw')

        objl = tkinter.Label(cf,text='objective:',anchor='e')
        objl.grid(row=11,column=0,sticky='e')
        rese = tkinter.Entry(cf,width=10,state='readonly',textvariable=self._vars['fit_control']['objective'])
        rese.grid(row=11,column=1,sticky='ew')
        gdfitcb = self.connected_checkbutton(cf,self._vars['fit_control']['good_fit'],self._set_good_fit,'Good fit')
        gdfitcb.grid(row=11,column=2,sticky='ew')

        cf.grid(row=1,pady=2,padx=2,sticky='ew')

    def _set_experiment_id(self,event=None):
        try:
            new_val = self._vars['fit_control']['experiment_id'].get()
        except:
            self._vars['fit_control']['experiment_id'].set(self.sys.sample_metadata['experiment_id'])
            new_val = self.sys.sample_metadata['experiment_id']
        if not new_val == self.sys.sample_metadata['experiment_id']:
            self.sys.sample_metadata['experiment_id'] = new_val
        return True

    def _set_sample_id(self,event=None):
        try:
            new_val = self._vars['fit_control']['sample_id'].get()
        except:
            self._vars['fit_control']['sample_id'].set(self.sys.sample_metadata['sample_id'])
            new_val = self.sys.sample_metadata['sample_id']
        if not new_val == self.sys.sample_metadata['sample_id']:
            self.sys.sample_metadata['sample_id'] = new_val
        return True

    def _set_wavelength(self,event=None):
        try:
            new_val = self._vars['fit_control']['wavelength'].get()
        except:
            self._vars['fit_control']['wavelength'].set(self.sys.sample_metadata['source_wavelength'])
            new_val = self.sys.sample_metadata['source_wavelength']
        if not new_val == self.sys.sample_metadata['source_wavelength']:
            self.sys.sample_metadata['source_wavelength'] = new_val
            self._draw_plots()
        return True

    def _set_q_range(self,q_idx,event=None):
        try:
            new_val = self._vars['fit_control']['q_range'][q_idx].get()
        except:
            self._vars['fit_control']['q_range'][q_idx].set(self.sys.fit_report['q_range'][q_idx])
            new_val = self.sys.fit_report['q_range'][q_idx]
        if not new_val == self.sys.fit_report['q_range'][q_idx]:
            self.sys.fit_report['q_range'][q_idx] = new_val
            self._update_fit_objective()
        return True

    def _set_error_weighted(self):
        new_val = self._vars['fit_control']['error_weighted'].get()
        if not new_val == self.sys.fit_report['error_weighted']:
            self.sys.fit_report['error_weighted'] = new_val
            self._update_fit_objective()
        return True

    def _set_logI_weighted(self):
        new_val = self._vars['fit_control']['logI_weighted'].get()
        if not new_val == self.sys.fit_report['logI_weighted']:
            self.sys.fit_report['logI_weighted'] = new_val
            self._update_fit_objective()
        return True

    def _set_good_fit(self):
        new_val = self._vars['fit_control']['good_fit'].get()
        self.sys.fit_report['good_fit'] = new_val

    def _create_noise_frame(self):
        nf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        nf.grid_columnconfigure(0,weight=1)
        self._frames['noise_model'] = nf
        nmf = tkinter.Frame(nf,bd=0) 
        nmf.grid_rowconfigure(0,minsize=30)
        #nmf.grid_columnconfigure(0,weight=1)
        nl = tkinter.Label(nmf,text='noise model:',width=12,anchor='e',padx=10)
        nl.grid(row=0,column=0,sticky='e')
        ntpvar = tkinter.StringVar(nmf)
        ntpvar.set(self.sys.noise_model.model)
        ntpvar.trace('w',self._update_noise)
        self._vars['noise_model'] = ntpvar
        ntpcb = ttk.Combobox(nmf,textvariable=ntpvar,
                state='readonly',values=xrsdefs.noise_model_names)
        ntpcb.grid(row=0,column=1,sticky='ew')
        nmf.grid(row=0,sticky='ew')

        self._frames['parameters']['noise'] = OrderedDict()
        self._vars['parameters']['noise'] = OrderedDict()
        for noise_param_nm in xrsdefs.noise_params[self.sys.noise_model.model]:
            self._frames['parameters']['noise'][noise_param_nm] = \
            self._create_param_frame('noise',noise_param_nm) 
        self._pack_noise_params()
        nf.grid(row=2,pady=2,padx=2,sticky='ew')

    def _repack_noise_frame(self):
        nmdl = self.sys.noise_model.model
        for par_nm,frm in self._frames['parameters']['noise'].items(): frm.grid_forget() 
        #self.fit_gui.update_idletasks()
        new_par_frms = OrderedDict()
        # NOTE: it is tempting to "keep" some frames that need not be renewed,
        # but if so, the widgets for the bounds, constraints, etc. would still need to be updated 
        # create new frames for all params
        for par_nm in xrsdefs.noise_params[nmdl]: 
            new_par_frms[par_nm] = self._create_param_frame('noise',par_nm)
        # destroy any frames that didn't get repacked
        par_frm_nms = list(self._frames['parameters']['noise'].keys())
        for par_nm in par_frm_nms: 
            if not par_nm in xrsdefs.noise_params[nmdl]: 
                frm = self._frames['parameters']['noise'].pop(par_nm)
                frm.destroy()
                self._vars['parameters']['noise'].pop(par_nm)
        # place the new frames in the _frames dict 
        self._frames['parameters']['noise'] = new_par_frms
        self._pack_noise_params()
        # update_idletasks() processes the frame changes, 
        # so that they are accounted for in control_canvas_configure()
        self.fit_gui.update_idletasks()
        self.control_canvas_configure()

    def _pack_noise_params(self):
        for param_idx, paramf in enumerate(self._frames['parameters']['noise'].values()):
            paramf.grid(row=1+param_idx,sticky='ew')

    def _create_pop_frame(self,pop_nm):
        pop = self.sys.populations[pop_nm]
        pf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        pf.grid_columnconfigure(0,weight=1)
        self._frames['populations'][pop_nm] = pf
        pop_struct = self.sys.populations[pop_nm].structure
        pop_form = self.sys.populations[pop_nm].form
        pop_settings = self.sys.populations[pop_nm].settings
        pop_params = self.sys.populations[pop_nm].parameters
        #
        # NAME, STRUCTURE, and FORM: 
        plf = tkinter.Frame(pf,bd=0)
        plf.grid_columnconfigure(2,weight=1)
        plf.grid_rowconfigure(1,minsize=30)
        plf.grid_rowconfigure(2,minsize=30)
        popl = tkinter.Label(plf,text='population:',anchor='e')
        popnml = tkinter.Label(plf,text=pop_nm,anchor='w')
        popl.grid(row=0,column=0,sticky='e')
        popnml.grid(row=0,column=1,padx=10,sticky='ew')
        rmb = tkinter.Button(plf,text='x',command=partial(self._remove_population,pop_nm))
        rmb.grid(row=0,column=2,sticky='e')
        #
        strl = tkinter.Label(plf,text='structure:',width=12,anchor='e')
        strl.grid(row=1,column=0,sticky='e')
        strvar = tkinter.StringVar(plf)
        strvar.set(pop_struct)
        strvar.trace('w',partial(self._update_structure,pop_nm))
        self._vars['structures'][pop_nm] = strvar
        strcb = ttk.Combobox(plf,textvariable=strvar,
                values=xrsdefs.structure_names,state='readonly')
        strcb.grid(row=1,column=1,sticky='ew')
        #
        ffl = tkinter.Label(plf,text='form factor:',width=12,anchor='e')
        ffl.grid(row=2,column=0,sticky='e')
        ffvar = tkinter.StringVar(plf)
        ffvar.set(pop_form)
        ffvar.trace('w',partial(self._update_form_factor,pop_nm))
        self._vars['form_factors'][pop_nm] = ffvar
        ffcb = ttk.Combobox(plf,textvariable=ffvar,
                values=xrsdefs.form_factor_names,state='readonly')
        ffcb.grid(row=2,column=1,sticky='ew') 
        plf.grid(row=0,sticky='ew')
        #
        # SETTINGS:
        self._frames['settings'][pop_nm] = OrderedDict()
        self._vars['settings'][pop_nm] = OrderedDict()
        for stg_nm in pop_settings:
            self._frames['settings'][pop_nm][stg_nm] = \
            self._create_setting_frame(pop_nm,stg_nm)
        #
        # PARAMETERS:
        self._frames['parameters'][pop_nm] = OrderedDict()
        self._vars['parameters'][pop_nm] = OrderedDict()
        for param_nm in pop_params:
            self._frames['parameters'][pop_nm][param_nm] = \
            self._create_param_frame(pop_nm,param_nm)
        #
        # PACKING:
        self._pack_setting_frames(pop_nm)
        self._pack_parameter_frames(pop_nm)
        #return pf

    def _repack_pop_frame(self,pop_nm):
        pop_struct = self.sys.populations[pop_nm].structure
        pop_form = self.sys.populations[pop_nm].form
        pop_settings = self.sys.populations[pop_nm].settings
        pop_params = self.sys.populations[pop_nm].parameters
        #
        # SETTINGS: 
        for stg_nm,frm in self._frames['settings'][pop_nm].items(): frm.grid_forget() 
        new_stg_frms = OrderedDict()
        # create new frames for all settings
        # NOTE: it is tempting to "keep" frames for settings that still hold,
        # but if so, the selection widgets would still need to be updated
        for stg_nm, stg_val in pop_settings.items():
            new_stg_frms[stg_nm] = self._create_setting_frame(pop_nm,stg_nm)
        # destroy any frames that didn't get repacked
        stg_frm_nms = list(self._frames['settings'][pop_nm].keys())
        for stg_nm in stg_frm_nms: 
            if not stg_nm in pop_settings: 
                frm = self._frames['settings'][pop_nm].pop(stg_nm)
                frm.destroy()
                self._vars['settings'][pop_nm].pop(stg_nm)
        # place the new frames in the _frames dict 
        self._frames['settings'][pop_nm] = new_stg_frms
        #
        # PARAMETERS: 
        for par_nm,frm in self._frames['parameters'][pop_nm].items(): frm.grid_forget() 
        #self.fit_gui.update_idletasks()
        new_par_frms = OrderedDict()
        # create new frames for all params
        # NOTE: it is tempting to "keep" some frames that need not be renewed,
        # but if so, the widgets for the bounds, constraints, etc. would still need to be updated 
        for par_nm in pop_params: 
            new_par_frms[par_nm] = self._create_param_frame(pop_nm,par_nm)
        # destroy any frames that didn't get repacked
        par_frm_nms = list(self._frames['parameters'][pop_nm].keys())
        for par_nm in par_frm_nms: 
            if not par_nm in pop_params: 
                frm = self._frames['parameters'][pop_nm].pop(par_nm)
                frm.destroy()
                self._vars['parameters'][pop_nm].pop(par_nm)
        # place the new frames in the _frames dict, repack 
        self._frames['parameters'][pop_nm] = new_par_frms
        self._pack_setting_frames(pop_nm)
        self._pack_parameter_frames(pop_nm)
        # update_idletasks() processes the frame changes, 
        # so that they are accounted for in control_canvas_configure()
        self.fit_gui.update_idletasks()
        self.control_canvas_configure()

    def _pack_population_frames(self):
        n_pop_frames = len(self._frames['populations'])
        for pop_idx, pop_nm in enumerate(self._frames['populations'].keys()):
            self._frames['populations'][pop_nm].grid(
            row=3+pop_idx,pady=2,padx=2,sticky='ew')
        self._frames['new_population'].grid(row=3+n_pop_frames,pady=2,padx=2,sticky='ew')

    def _repack_pop_frames(self):
        for pop_nm,frm in self._frames['populations'].items(): frm.grid_forget() 
        self._frames['new_population'].grid_forget()
        for pop_nm in self.sys.populations.keys():
            if not pop_nm in self._frames['populations']:
                self._create_pop_frame(pop_nm)
        pop_frm_nms = list(self._frames['populations'].keys())
        for pop_nm in pop_frm_nms:
            if not pop_nm in self.sys.populations: 
                frm = self._frames['populations'].pop(pop_nm)
                frm.destroy()
                # TODO (low): clean up refs to obsolete vars
                # and widgets that were children of this frame 
        self._pack_population_frames()
        # update_idletasks() processes the frame changes, 
        # so that they are accounted for in control_canvas_configure()
        self.fit_gui.update_idletasks()
        self.control_canvas_configure()

    def _pack_setting_frames(self,pop_nm):
        for stg_idx, stg_frm in enumerate(self._frames['settings'][pop_nm].values()):
            stg_frm.grid(row=1+stg_idx,sticky='ew')

    def _pack_parameter_frames(self,pop_nm):
        n_stg_frms = len(self._frames['settings'][pop_nm])
        for param_idx, paramf in enumerate(self._frames['parameters'][pop_nm].values()):
            paramf.grid(row=1+n_stg_frms+param_idx,sticky='ew')

    def _create_setting_frame(self,pop_nm,stg_nm):
        stg_vars = self._vars['settings'][pop_nm]
        stg_frames = self._frames['settings'][pop_nm]
        parent_obj = self.sys.populations[pop_nm]
        parent_frame = self._frames['populations'][pop_nm]
        stgf = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        stgf.grid_columnconfigure(1,weight=1)

        if xrsdefs.setting_datatypes(stg_nm) is str:
            stgv = tkinter.StringVar(parent_frame)
        elif xrsdefs.setting_datatypes(stg_nm) is int:
            stgv = tkinter.IntVar(parent_frame)
        elif xrsdefs.setting_datatypes(stg_nm) is float:
            stgv = tkinter.DoubleVar(parent_frame)
        elif xrsdefs.setting_datatypes(stg_nm) is bool:
            stgv = tkinter.BooleanVar(parent_frame)
        stg_frames[stg_nm] = stgf
        stg_vars[stg_nm] = stgv

        stgl = tkinter.Label(stgf,text='{}:'.format(stg_nm),width=18,anchor='e')
        stgl.grid(row=0,column=0,sticky='e')
        s = parent_obj.settings[stg_nm]
        stgv.set(str(s))

        stg_sel = xrsdefs.setting_selections(stg_nm,parent_obj.structure,parent_obj.form,parent_obj.settings)
        if stg_sel:
            stgcb = ttk.Combobox(stgf,textvariable=stgv,
                    values=stg_sel,state='readonly')
            stgcb.grid(row=0,column=1,sticky='ew')
            stgv.trace('w',partial(self._update_setting,pop_nm,stg_nm))
        else:
            stge = self.connected_entry(stgf,stgv,partial(self._update_setting,pop_nm,stg_nm))
            stge.grid(row=0,column=1,sticky='w')
        return stgf

    def _create_param_frame(self,pop_nm,param_nm):
        param_vars = self._vars['parameters'][pop_nm]
        param_frames = self._frames['parameters'][pop_nm]
        param_var_nm = pop_nm+'__'+param_nm
        if pop_nm == 'noise': 
            parent_frame = self._frames['noise_model']
            parent_obj = self.sys.noise_model
        else:
            parent_frame = self._frames['populations'][pop_nm]
            parent_obj = self.sys.populations[pop_nm]
        param_def = parent_obj.parameters[param_nm]
        param_idx = len(param_frames)
        if not param_nm in param_vars: param_vars[param_nm] = {}

        paramf = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        paramf.grid_columnconfigure(1,weight=1)
        paramf.grid_columnconfigure(2,weight=1)
        paramv = tkinter.DoubleVar(paramf)
        param_frames[param_nm] = paramf
        param_vars[param_nm]['value'] = paramv

        pl = tkinter.Label(paramf,text='parameter:',anchor='e')
        pl.grid(row=0,column=0,sticky='e')
        pnml = tkinter.Label(paramf,text=param_nm,anchor='w') 
        pnml.grid(row=0,column=1,sticky='w')

        pfixvar = tkinter.BooleanVar(paramf)
        param_vars[param_nm]['fixed'] = pfixvar
        pfixvar.set(param_def['fixed'])
        psw = self.connected_checkbutton(paramf,pfixvar,
            partial(self._update_param,pop_nm,param_nm,'fixed'),'fixed')
        psw.grid(row=0,column=2,sticky='w')

        vl = tkinter.Label(paramf,text='value:',anchor='e')
        vl.grid(row=1,column=0,columnspan=1,sticky='e')
        paramv.set(param_def['value'])
        pe = self.connected_entry(paramf,paramv,
            partial(self._update_param,pop_nm,param_nm,'value'),16)
        pe.grid(row=1,column=1,columnspan=2,sticky='ew')

        pbndl = tkinter.Label(paramf,text='bounds:',anchor='e')
        pbndl.grid(row=2,column=0,sticky='e')
        lbndv = tkinter.StringVar(paramf)
        ubndv = tkinter.StringVar(paramf)
        param_vars[param_nm]['bounds']=[lbndv,ubndv]
        pbnde1 = self.connected_entry(paramf,lbndv,
            partial(self._update_param,pop_nm,param_nm,'bounds',0),8)
        lbndv.set(param_def['bounds'][0])
        ubndv.set(param_def['bounds'][1])
        pbnde2 = self.connected_entry(paramf,ubndv,
            partial(self._update_param,pop_nm,param_nm,'bounds',1),8)
        pbnde1.grid(row=2,column=1,sticky='ew')
        pbnde2.grid(row=2,column=2,sticky='ew')

        pvarl = tkinter.Label(paramf,text='variable name:',anchor='e')
        pvarl.grid(row=3,column=0,sticky='e')
        pvar = tkinter.Entry(paramf,width=18) 
        pvar.insert(0,param_var_nm) 
        pvar.config(state='readonly')
        pvar.grid(row=3,column=1,columnspan=2,sticky='ew')

        pexpl = tkinter.Label(paramf,text='constraint:',anchor='e')
        pexpl.grid(row=4,column=0,sticky='e')
        exprv = tkinter.StringVar(paramf)
        param_vars[param_nm]['constraint_expr'] = exprv 
        exprv.set(param_def['constraint_expr'])
        pexpe = self.connected_entry(paramf,exprv,
            partial(self._update_param,pop_nm,param_nm,'constraint_expr'),16)
        pexpe.grid(row=4,column=1,columnspan=2,sticky='ew')
        return paramf

    def _create_new_pop_frame(self):
        npf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        npf.grid_columnconfigure(1,weight=1)
        self._frames['new_population'] = npf
        addl = tkinter.Label(npf,text='new population:',anchor='w')
        addl.grid(row=0,column=0,sticky='w')
        self._vars['new_population_name'] = tkinter.StringVar(npf)
        nme = self.connected_entry(npf,self._vars['new_population_name'],None,12)
        nme.grid(row=0,column=1,sticky='ew')
        nme.bind('<Return>',self._new_population)
        addb = tkinter.Button(npf,text='+',command=self._new_population)
        addb.grid(row=0,column=2,sticky='e')
        npops = len(self._frames['populations'])
        #return npf

    def _draw_plots(self):
        I_comp = draw_xrsd_fit(self.fig.gca(),self.sys,self.q,self.I,self.dI)
        self.mpl_canvas.draw()
        self._update_fit_objective(I_comp)

    def _update_fit_objective(self,I_comp=None):
        obj_val = self.sys.evaluate_residual(
            self.q,self.I,self.dI,I_comp)
        self._vars['fit_control']['objective'].set(str(obj_val))

    def _update_param(self,pop_nm,param_nm,param_key,param_idx=None,event=None):
        # param_key should be 'value', 'fixed', 'bounds', or 'constraint_expr'
        # if param_key == 'bounds', param_idx must be 0 or 1
        vflag = self._validate_param(pop_nm,param_nm,param_key,param_idx)
        if vflag:
            if pop_nm == 'noise':
                x = self.sys.noise_model
            else: 
                x = self.sys.populations[pop_nm]
            tkv = self._vars['parameters'][pop_nm][param_nm]
            xp = x.parameters[param_nm]
            new_param = copy.deepcopy(xp)
            param_changed = False
            if param_idx in [0,1]: 
                new_val = tkv[param_key][param_idx].get()
                if new_val in ['None','none','']:
                    new_param[param_key][param_idx] = None
                else: 
                    new_param[param_key][param_idx] = float(new_val) 
                if not new_param[param_key][param_idx] == xp[param_key][param_idx]: 
                    param_changed = True
            else:
                new_val = tkv[param_key].get()
                if param_key == 'constraint_expr':
                    if new_val in ['None','none','']:
                        new_val = None
                new_param[param_key] = new_val 
                if not new_param[param_key] == xp[param_key]: 
                    param_changed = True
            if param_changed: 
                x.update_parameters({param_nm:new_param})
                self._draw_plots()
        return vflag

    def _update_setting(self,pop_nm,stg_nm,*event_args):
        vflag = self._validate_setting(pop_nm,stg_nm) 
        if vflag:
            x = self.sys.populations[pop_nm]
            tkv = self._vars['settings'][pop_nm][stg_nm]
            new_val = tkv.get()
            if not new_val == x.settings[stg_nm]:
                x.update_settings({stg_nm:new_val})
                self._repack_pop_frame(pop_nm)
                self._draw_plots()
        return vflag

    def _validate_setting(self,pop_nm,stg_nm):
        """Validate a setting Var entry and set its value in self.sys"""
        x = self.sys.populations[pop_nm].settings[stg_nm]
        tkv = self._vars['settings'][pop_nm][stg_nm]
        is_valid = True
        try:
            new_val = tkv.get()
        except:
            is_valid = False
            tkv.set(x)
        return is_valid

    def _validate_param(self,pop_nm,param_nm,param_key,param_idx=None):
        """Validate a parameter Var entry and set its value in self.sys 

        If the entry is valid, the Variable is set to the Entry's value. 
        If the entry is not valid, the Variable is reset.
        
        Parameters
        ----------
        pop_nm : string 
        param_nm : string 
        param_key : string 
        param_idx : int 

        Returns
        -------
        is_valid : boolean
            Flag for whether or not the entry was found to be valid
        """
        if pop_nm == 'noise':
            x = self.sys.noise_model.parameters[param_nm]
            tkvs = self._vars['parameters'][pop_nm][param_nm]
        else:
            x = self.sys.populations[pop_nm].parameters[param_nm]
            tkvs = self._vars['parameters'][pop_nm][param_nm]
        is_valid = True
        if param_idx in [0,1]:
            old_val = x[param_key][param_idx]
            try:
                new_val = tkvs[param_key][param_idx].get()
                if not new_val in ['None','none','']:
                    new_val = float(new_val)
            except:
                is_valid = False
                tkvs[param_key][param_idx].set(old_val)
        else:
            old_val = x[param_key]
            try:
                new_val = tkvs[param_key].get()
            except:
                is_valid = False
                tkvs[param_key].set(old_val)
        return is_valid

    def _new_population(self,event=None):
        new_nm = self._vars['new_population_name'].get()
        if new_nm and not new_nm in self.sys.populations:
            self.sys.add_population(new_nm,'diffuse','atomic')
            self._frames['new_population'].grid_forget() 
            self._create_pop_frame(new_nm)
            npops = len(self._frames['populations'])
            self._frames['populations'][new_nm].grid(row=2+npops,padx=2,pady=2,sticky='ew') 
            self._frames['new_population'].grid(row=3+npops,padx=2,pady=2,sticky='ew') 
            # update_idletasks() processes the new frame,
            # so that it is accounted for in control_canvas_configure()
            self.fit_gui.update_idletasks()
            self.control_canvas_configure()

    def _update_noise(self,*event_args):
        s = self._vars['noise_model'].get()
        if not s == self.sys.noise_model.model:
            try:
                self.sys.noise_model.set_model(s)
            except:
                raise
            self._repack_noise_frame()
            self._draw_plots()

    def _update_structure(self,pop_nm,*event_args):
        s = self._vars['structures'][pop_nm].get()
        if not s == self.sys.populations[pop_nm].structure:
            try:
                self.sys.populations[pop_nm].set_structure(s)
            except:
                self._vars['structures'][pop_nm].set(self.sys.populations[pop_nm].structure)
            self._repack_pop_frame(pop_nm)
            self._draw_plots()

    def _update_form_factor(self,pop_nm,*event_args):
        f = self._vars['form_factors'][pop_nm].get()
        if not f == self.sys.populations[pop_nm].form:
            try:
                self.sys.populations[pop_nm].set_form(f)
            except:
                self._vars['form_factors'][pop_nm].set(
                self.sys.populations[pop_nm].form)
            self._repack_pop_frame(pop_nm)
            self._draw_plots()

    def _remove_population(self,pop_nm):
        self.sys.remove_population(pop_nm)
        self._repack_pop_frames()
        self._draw_plots()

    def _fit(self):
        sys_opt = xrsdsys.fit(self.sys,self.q,self.I,self.dI)
        self.sys.update_from_dict(sys_opt.to_dict())
        self._update_parameter_values() 
        self._draw_plots()

    def _update_parameter_values(self):
        for param_nm,par in self.sys.noise_model.parameters.items():
            self._vars['parameters']['noise'][param_nm]['value'].set(par['value'])
        for pop_nm,pop in self.sys.populations.items():        
            for param_nm in pop.parameters.keys():
                self._vars['parameters'][pop_nm][param_nm]['value'].set(
                pop.parameters[param_nm]['value'])

    def _estimate(self):
        feats = profiler.profile_pattern(self.q,self.I)
        pred = xrsdpred.predict(feats)
        sys_est = xrsdpred.system_from_prediction(pred,self.q,self.I,
            features = self.sys.features,
            sample_metadata = self.sys.sample_metadata,
            fit_report = self.sys.fit_report
            )
        self._set_system(sys_est)

    def _set_system(self,new_sys):
        self.sys = new_sys 
        # update fit control widgets for new system
        self._vars['fit_control']['good_fit'].set(self.sys.fit_report['good_fit'])
        self._vars['fit_control']['q_range'][0].set(self.sys.fit_report['q_range'][0])
        self._vars['fit_control']['q_range'][1].set(self.sys.fit_report['q_range'][1])
        self._vars['fit_control']['error_weighted'].set(self.sys.fit_report['error_weighted'])
        self._vars['fit_control']['logI_weighted'].set(self.sys.fit_report['logI_weighted'])
        self._vars['fit_control']['experiment_id'].set(self.sys.sample_metadata['experiment_id'])
        self._vars['fit_control']['sample_id'].set(self.sys.sample_metadata['sample_id'])
        if any([pp.structure=='crystalline' for pp in new_sys.populations.values()]):
            if new_sys.sample_metadata['source_wavelength'] == 0.:
                # TODO: put this warning in a pop-up
                warnings.warn('Diffraction computations require a nonzero wavelength: setting default wavelength==1.')
                new_sys.sample_metadata['source_wavelength'] = 1.
        self._vars['fit_control']['wavelength'].set(self.sys.sample_metadata['source_wavelength'])
        # repack everything 
        self._repack_noise_frame()
        self._repack_pop_frames()
        for pop_nm in self.sys.populations.keys():
            self._repack_pop_frame(pop_nm)
        # draw plots and update fit objective
        self._draw_plots()

    @staticmethod
    def on_mousewheel(canvas,event):
        canvas.yview_scroll(-1*event.delta,'units')

    @staticmethod
    def on_trackpad(canvas,event):
        if event.num == 4:
            d = -2
        elif event.num == 5:
            d = 2
        canvas.yview_scroll(d,'units')

    @staticmethod
    def connected_entry(parent,tkvar,cbfun=None,entry_width=None):
        if cbfun:
            # piggyback on entry validation to update internal data
            # NOTE: validatecommand must return a boolean, or else it will disconnect quietly
            if entry_width:
                e = tkinter.Entry(parent,width=entry_width,textvariable=tkvar,validate="focusout",validatecommand=cbfun)
            else:
                e = tkinter.Entry(parent,textvariable=tkvar,validate="focusout",validatecommand=cbfun)
            # also respond to the return key
            e.bind('<Return>',cbfun)
        else:
            if entry_width:
                e = tkinter.Entry(parent,width=entry_width,textvariable=tkvar)
            else:
                e = tkinter.Entry(parent,textvariable=tkvar)
        return e

    @staticmethod
    def connected_checkbutton(parent,boolvar,cbfun=None,label=''):
        if cbfun:
            e = tkinter.Checkbutton(parent,text=label,variable=boolvar,command=cbfun,anchor='w')
        else:
            e = tkinter.Checkbutton(parent,text=label,variable=boolvar,anchor='w')
        return e

