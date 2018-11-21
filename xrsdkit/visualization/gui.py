from collections import OrderedDict
from functools import partial
import copy
import sys
import os

import numpy as np
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use("TkAgg")
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
from ..system.population import StructureFormException 

if sys.version_info[0] < 3:
    import Tkinter as tkinter
else:
    import tkinter

def run_fit_gui(system,q,I,source_wavelength,
    dI=None,error_weighted=True,
    logI_weighted=True,
    q_range=[0.,float('inf')],
    good_fit_prior=False):
    gui = XRSDFitGUI(system,q,I,source_wavelength,dI,error_weighted,logI_weighted,q_range,good_fit_prior)
    sys_opt, good_fit = gui.start()
    # collect results and return
    return sys_opt, good_fit

# TODO (low): when a structure or form selection is rejected,
#   get the associated combobox re-painted-
#   currently the value does get reset, 
#   but the cb does not repaint until it is focused-on

# TODO (low): in _validate_param(), 
#   if param_key == 'constraint_expr', 
#   validate the expression with lmfit/asteval

# TODO (low): when a param is fixed or has a constraint set,
#   make the entry widget read-only

class XRSDFitGUI(object):

    def __init__(self,system,
        q,I,source_wavelength,
        dI=None,error_weighted=True,
        logI_weighted=True,
        q_range=[0.,float('inf')],
        good_fit_prior=False):

        super(XRSDFitGUI, self).__init__()
        self.q = q
        self.I = I
        self.dI = dI
        self.src_wl = source_wavelength
        if not system: system = xrsdsys.System()
        self.sys = system
        self.error_weighted = error_weighted
        self.logI_weighted = logI_weighted
        self.q_range = q_range
        self.good_fit = good_fit_prior

        self.fit_gui = tkinter.Tk()
        self.fit_gui.protocol('WM_DELETE_WINDOW',self._cleanup)
        # setup the main gui objects
        self._build_gui()
        # create the widgets for control 
        self._build_control_widgets()
        # create the plots
        self._build_plot_widgets()
        # draw the plots...
        #self._draw_plots()
        self.fit_gui.geometry('1100x700')

    def start(self):
        # start the tk loop
        self.fit_gui.mainloop()
        # after the loop, return the (optimized) system
        return self.sys, self.good_fit

    def _cleanup(self):
        # remove references to all gui objects, widgets, etc. 
        #self._reset_control_widgets() 
        self.fit_gui.quit()
        self.fit_gui.destroy()

    def _build_gui(self):
        self.fit_gui.title('xrsd profile fitter')
        # a horizontal scrollbar and a main canvas belong to the main gui:
        scrollbar = tkinter.Scrollbar(self.fit_gui,orient='horizontal')
        main_canvas = tkinter.Canvas(self.fit_gui)
        scrollbar.pack(side=tkinter.BOTTOM,fill=tkinter.X)
        main_canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)
        scrollbar.config(command=main_canvas.xview)
        main_canvas.config(xscrollcommand=scrollbar.set)
        # the main widget will be a frame,
        # displayed as a window item on the main canvas:
        self.main_frame = tkinter.Frame(main_canvas,bd=4,relief=tkinter.SUNKEN)
        main_frame_window = main_canvas.create_window(0,0,window=self.main_frame,anchor='nw')
        # _canvas_configure() ensures that the window item and scrollbar
        # remain the correct size for the underlying widget
        self.main_canvas_configure = partial(self._canvas_configure,main_canvas,self.main_frame,main_frame_window)  
        main_canvas.bind("<Configure>",self.main_canvas_configure)

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
        # the main frame contains a plot frame on the left,
        # containing a canvas, which contains a window item,
        # which displays a view on a plot widget 
        # built from FigureCanvasTkAgg.get_tk_widget()
        plot_frame = tkinter.Frame(self.main_frame,bd=4,relief=tkinter.SUNKEN)
        plot_frame.pack(side=tkinter.LEFT,fill=tkinter.BOTH,expand=True,padx=2,pady=2)
        self.fig,I_comp = plot_xrsd_fit(self.sys,self.q,self.I,self.src_wl,self.dI,False)
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
        self._update_fit_objective(I_comp)

    def _build_control_widgets(self):
        # the main frame contains a control frame on the right,
        # containing a canvas, which contains a window item,
        # which displays a view on a frame full of entry widgets and labels, 
        # which are used to control parameters, settings, etc. 
        control_frame = tkinter.Frame(self.main_frame)
        control_frame.pack(side=tkinter.RIGHT,fill='y')
        control_frame_canvas = tkinter.Canvas(control_frame)
        control_frame.bind_all("<MouseWheel>", partial(self.on_mousewheel,control_frame_canvas))
        control_frame.bind_all("<Button-4>", partial(self.on_trackpad,control_frame_canvas))
        control_frame.bind_all("<Button-5>", partial(self.on_trackpad,control_frame_canvas))
        yscr = tkinter.Scrollbar(control_frame)
        yscr.pack(side=tkinter.RIGHT,fill='y')
        control_frame_canvas.pack(fill='both',expand=True)
        control_frame_canvas.config(yscrollcommand=yscr.set)
        yscr.config(command=control_frame_canvas.yview)
        # TODO (low): figure out a way to set or control the width of the control widget
        # NOTE: currently it takes on the net width of the entry widgets
        self.control_widget = tkinter.Frame(control_frame_canvas)
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
            species=OrderedDict(),
            parameters=OrderedDict(),
            settings=OrderedDict(),
            specie_parameters=OrderedDict(),
            specie_settings=OrderedDict(),
            new_population=None,
            new_species=OrderedDict(),
            fit_control=OrderedDict()
            )
        self._vars = OrderedDict(
            noise_model=None,
            structures=OrderedDict(),
            form_factors=OrderedDict(),
            specie_parameters=OrderedDict(),
            specie_settings=OrderedDict(),
            parameters=OrderedDict(),
            settings=OrderedDict(),
            new_population_name=None,
            new_specie_names=OrderedDict(),
            fit_control=OrderedDict()
            )

    def _create_control_widgets(self):
        self._frames['parameters']['noise'] = OrderedDict()
        self._vars['parameters']['noise'] = OrderedDict()
        self._create_fit_control_frame()
        self._create_noise_frame()
        for pop_nm in self.sys.populations.keys():
            self._frames['parameters'][pop_nm] = OrderedDict()
            self._vars['parameters'][pop_nm] = OrderedDict()
            self._frames['populations'][pop_nm] = self._create_pop_frame(pop_nm)
        self._frames['new_population'] = self._create_new_pop_frame()
        self._pack_population_frames()

    def _pack_population_frames(self):
        n_pop_frames = len(self._frames['populations'])
        for pop_idx, pop_nm in enumerate(self._frames['populations'].keys()):
            self._frames['populations'][pop_nm].grid(
            row=2+pop_idx,pady=2,padx=2,sticky='ew')
        self._frames['new_population'].grid(row=2+n_pop_frames,pady=2,padx=2,sticky='ew')

    def _repack_pop_frames(self):
        for pop_nm,frm in self._frames['populations'].items(): frm.pack_forget() 
        self._frames['new_population'].pack_forget()
        new_pop_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        for pop_nm in self._frames['populations'].keys(): 
            if pop_nm in self.sys.populations:
                new_pop_frms[pop_nm] = self._frames['populations'][pop_nm]
        for pop_nm in self.sys.populations.keys():
            if not pop_nm in self._frames['populations']:
                new_pop_frms[pop_nm] = self._create_pop_frame(pop_nm)
        # destroy any frames that didn't get saved
        pop_frm_nms = list(self._frames['populations'].keys())
        for pop_nm in pop_frm_nms:
            if not pop_nm in self.sys.populations: 
                frm = self._frames['populations'].pop(pop_nm)
                frm.destroy()
                # TODO (low): clean up refs to obsolete vars
                # and widgets that were children of this frame 
        # place the new frames in the _frames dict 
        self._frames['populations'] = new_pop_frms
        self._pack_population_frames()
        # update_idletasks() processes the frame changes, 
        # so that they are accounted for in control_canvas_configure()
        self.fit_gui.update_idletasks()
        self.control_canvas_configure()

    def _create_fit_control_frame(self):
        # TODO: file io q,I (dat/csv) and populations (YAML); output for DB records (JSON)

        cf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        cf.grid_columnconfigure(1,weight=1)
        cf.grid_columnconfigure(2,weight=1)
        self._frames['fit_control'] = cf
        self._vars['fit_control']['wavelength'] = tkinter.DoubleVar(cf)
        self._vars['fit_control']['wavelength'].set(self.src_wl)
        self._vars['fit_control']['objective'] = tkinter.StringVar(cf)
        self._vars['fit_control']['error_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['logI_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['error_weighted'].set(self.error_weighted)
        self._vars['fit_control']['logI_weighted'].set(self.logI_weighted)
        self._vars['fit_control']['q_range'] = [tkinter.DoubleVar(cf),tkinter.DoubleVar(cf)]
        self._vars['fit_control']['q_range'][0].set(self.q_range[0])
        self._vars['fit_control']['q_range'][1].set(self.q_range[1])
        self._vars['fit_control']['good_fit'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['good_fit'].set(self.good_fit)
        
        wll = tkinter.Label(cf,text='wavelength:',anchor='e')
        wle = self.connected_entry(cf,self._vars['fit_control']['wavelength'],self._set_wavelength,10)
        wll.grid(row=0,column=0,sticky='e')
        wle.grid(row=0,column=1,sticky='ew')

        q_range_lbl = tkinter.Label(cf,text='q-range:',anchor='e')
        q_range_lbl.grid(row=1,column=0,sticky='e')
        #q_lo_ent = tkinter.Entry(cf,width=8,textvariable=self._vars['fit_control']['q_range'][0])
        #q_hi_ent = tkinter.Entry(cf,width=8,textvariable=self._vars['fit_control']['q_range'][1])
        q_lo_ent = self.connected_entry(cf,self._vars['fit_control']['q_range'][0],
            partial(self._set_q_range,0),8) 
        q_hi_ent = self.connected_entry(cf,self._vars['fit_control']['q_range'][1],
            partial(self._set_q_range,1),8) 
        q_lo_ent.grid(row=1,column=1,sticky='ew')
        q_hi_ent.grid(row=1,column=2,sticky='ew')

        ewtcb = self.connected_checkbutton(cf,self._vars['fit_control']['error_weighted'],
            self._set_error_weighted,'error weighted')
        ewtcb.grid(row=2,column=0,sticky='w')
        logwtcb = self.connected_checkbutton(cf,self._vars['fit_control']['logI_weighted'],
            self._set_logI_weighted,'log(I) weighted')
        logwtcb.grid(row=3,column=0,sticky='w')

        estbtn = tkinter.Button(cf,text='Estimate',width=8,command=self._estimate)
        estbtn.grid(row=2,column=1,rowspan=2,sticky='nesw')
        fitbtn = tkinter.Button(cf,text='Fit',width=8,command=self._fit)
        fitbtn.grid(row=2,column=2,rowspan=2,sticky='nesw')

        objl = tkinter.Label(cf,text='objective:',anchor='e')
        objl.grid(row=4,column=0,sticky='e')
        rese = tkinter.Entry(cf,width=10,state='readonly',textvariable=self._vars['fit_control']['objective'])
        rese.grid(row=4,column=1,sticky='ew')
        fitcb = self.connected_checkbutton(cf,self._vars['fit_control']['good_fit'],
            self._set_good_fit,'Good fit')
        fitcb.grid(row=4,column=2,sticky='ew')

        cf.grid(row=0,pady=2,padx=2,sticky='ew')

    def _set_wavelength(self,event=None):
        try:
            new_val = self._vars['fit_control']['wavelength'].get()
        except:
            self._vars['fit_control']['wavelength'].set(self.src_wl)
            new_val = self.src_wl
            return False
        if not new_val == self.src_wl:
            self.src_wl = new_val
            self._draw_plots()
        return True

    def _set_q_range(self,q_idx,event=None):
        try:
            new_val = self._vars['fit_control']['q_range'][q_idx].get()
        except:
            self._vars['fit_control']['q_range'][q_idx].set(self.q_range[q_idx])
            new_val = self.q_range[q_idx]
            return False
        if not new_val == self.q_range[q_idx]:
            self.q_range[q_idx] = new_val
            self._update_fit_objective()
        return True

    def _set_error_weighted(self):
        new_val = self._vars['fit_control']['error_weighted'].get()
        if not new_val == self.error_weighted:
            self.error_weighted = new_val
            self._update_fit_objective()
        return True

    def _set_logI_weighted(self):
        new_val = self._vars['fit_control']['logI_weighted'].get()
        if not new_val == self.logI_weighted:
            self.logI_weighted = new_val
            self._update_fit_objective()
        return True

    def _set_good_fit(self):
        new_val = self._vars['fit_control']['good_fit'].get()
        self.good_fit = new_val

    def _create_noise_frame(self):
        nf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        self._frames['noise_model'] = nf
        nmf = tkinter.Frame(nf,bd=0) 
        nl = tkinter.Label(nmf,text='noise model:',width=12,anchor='e',padx=10)
        nl.pack(side=tkinter.LEFT)
        ntpvar = tkinter.StringVar(nmf)
        ntp_option_dict = list(xrsdefs.noise_model_names)
        ntpcb = tkinter.OptionMenu(nmf,ntpvar,*ntp_option_dict)
        ntpvar.set(self.sys.noise_model.model)
        ntpvar.trace('w',self._repack_noise_frame)
        ntpcb.pack(side=tkinter.LEFT,fill='x')
        self._vars['noise_model'] = ntpvar
        nmf.grid(row=0,sticky='ew')

        self._frames['parameters']['noise'] = OrderedDict()
        self._vars['parameters']['noise'] = OrderedDict()
        for noise_param_nm in xrsdefs.noise_params[self.sys.noise_model.model]:
            self._frames['parameters']['noise'][noise_param_nm] = \
            self._create_param_frame('noise',None,noise_param_nm) 
        self._pack_noise_params()
        nf.grid(row=1,pady=2,padx=2,sticky='ew')

    def _repack_noise_frame(self):
        nmdl = self.sys.noise_model.model
        for par_nm,frm in self._frames['parameters']['noise'].items(): frm.pack_forget() 
        new_par_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        for par_nm in xrsdefs.noise_model_params[nmdl]: 
            if par_nm in self._frames['parameters']['noise']:
                new_par_frms[par_nm] = self._frames['parameters']['noise'][par_nm]
            else:
                new_par_frms[par_nm] = self._create_param_frame('noise',None,par_nm)
        # destroy any frames that didn't get repacked
        par_frm_nms = list(self._frames['parameters']['noise'].keys())
        for par_nm in par_frm_nms: 
            if not par_nm in xrsdefs.noise_model_params[nmdl]: 
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
        self._frames['populations'][pop_nm] = pf
        pop_struct = self.sys.populations[pop_nm].structure
        #
        # NAME and STRUCTURE: 
        plf = tkinter.Frame(pf,bd=0)
        plf.grid_columnconfigure(2,weight=1)
        popl = tkinter.Label(plf,text='population:',anchor='e')
        popnml = tkinter.Label(plf,text=pop_nm,anchor='w')
        popl.grid(row=0,column=0,sticky='e')
        popnml.grid(row=0,column=1,padx=10,sticky='ew')
        rmb = tkinter.Button(plf,text='x',command=partial(self._remove_population,pop_nm))
        rmb.grid(row=0,column=2,sticky='e')
        strl = tkinter.Label(plf,text='structure:',width=12,anchor='e')
        strl.grid(row=1,column=0,sticky='e')
        strvar = tkinter.StringVar(plf)
        str_option_dict = OrderedDict.fromkeys(xrsdefs.structure_names)
        strcb = tkinter.OptionMenu(plf,strvar,*str_option_dict)
        strvar.set(pop_struct)
        strvar.trace('w',partial(self._update_structure,pop_nm))
        strcb.grid(row=1,column=1,sticky='ew')
        self._vars['structures'][pop_nm] = strvar
        plf.grid(row=0,sticky='ew')
        #
        # SETTINGS:
        self._frames['settings'][pop_nm] = OrderedDict()
        self._vars['settings'][pop_nm] = OrderedDict()
        for stg_nm in xrsdefs.structure_settings[pop_struct]:
            self._frames['settings'][pop_nm][stg_nm] = \
            self._create_setting_frame(pop_nm,None,stg_nm)
        #
        # PARAMETERS:
        self._frames['parameters'][pop_nm] = OrderedDict()
        self._vars['parameters'][pop_nm] = OrderedDict()
        param_nms = copy.deepcopy(xrsdefs.structure_params[pop_struct])
        if pop_struct == 'crystalline':
            pop_lat = self.sys.populations[pop_nm].settings['lattice'] 
            param_nms.extend(copy.deepcopy(xrsdefs.setting_params['lattice'][pop_lat]))
        if pop_struct == 'disordered':
            pop_interxn = self.sys.populations[pop_nm].settings['interaction'] 
            param_nms.extend(copy.deepcopy(xrsdefs.setting_params['interaction'][pop_interxn]))
        for param_nm in param_nms:
            self._frames['parameters'][pop_nm][param_nm] = \
            self._create_param_frame(pop_nm,None,param_nm)
        #
        # SPECIES:
        self._frames['species'][pop_nm] = OrderedDict()
        self._frames['specie_parameters'][pop_nm] = OrderedDict()
        self._frames['specie_settings'][pop_nm] = OrderedDict()
        self._vars['form_factors'][pop_nm] = OrderedDict()
        self._vars['specie_parameters'][pop_nm] = OrderedDict()
        self._vars['specie_settings'][pop_nm] = OrderedDict()
        for specie_nm,specie in self.sys.populations[pop_nm].basis.items():
            self._frames['species'][pop_nm][specie_nm] = \
            self._create_specie_frame(pop_nm,specie_nm)
        self._frames['new_species'][pop_nm] = self._create_new_specie_frame(pop_nm)
        #
        # PACKING:
        self._pack_setting_frames(pop_nm)
        self._pack_parameter_frames(pop_nm)
        self._pack_specie_frames(pop_nm)
        return pf

    def _repack_pop_frame(self,pop_nm):
        pop_struct = self.sys.populations[pop_nm].structure
        #
        # SETTINGS: 
        for stg_nm,frm in self._frames['settings'][pop_nm].items(): frm.pack_forget() 
        new_stg_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        for stg_nm in xrsdefs.structure_settings[pop_struct]:
            if stg_nm in self._frames['settings'][pop_nm]:
                new_stg_frms[stg_nm] = self._frames['settings'][pop_nm][stg_nm]
            else:
                new_stg_frms[stg_nm] = self._create_setting_frame(pop_nm,None,stg_nm)
        # destroy any frames that didn't get repacked
        stg_frm_nms = list(self._frames['settings'][pop_nm].keys())
        for stg_nm in stg_frm_nms: 
            if not stg_nm in xrsdefs.structure_settings[pop_struct]: 
                frm = self._frames['settings'][pop_nm].pop(stg_nm)
                frm.destroy()
                self._vars['settings'][pop_nm].pop(stg_nm)
        # place the new frames in the _frames dict 
        self._frames['settings'][pop_nm] = new_stg_frms
        #
        # PARAMETERS: 
        for par_nm,frm in self._frames['parameters'][pop_nm].items(): frm.pack_forget() 
        new_par_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        param_nms = copy.deepcopy(xrsdefs.structure_params[pop_struct])
        if pop_struct == 'crystalline':
            pop_lat = self.sys.populations[pop_nm].settings['lattice'] 
            param_nms.extend(copy.deepcopy(xrsdefs.setting_params['lattice'][pop_lat]))
        if pop_struct == 'disordered':
            pop_interxn = self.sys.populations[pop_nm].settings['interaction'] 
            param_nms.extend(copy.deepcopy(xrsdefs.setting_params['interaction'][pop_interxn]))
        for par_nm in param_nms: 
            if par_nm in self._frames['parameters'][pop_nm]:
                new_par_frms[par_nm] = self._frames['parameters'][pop_nm][par_nm]
            else:
                new_par_frms[par_nm] = self._create_param_frame(pop_nm,None,par_nm)
        # destroy any frames that didn't get repacked
        par_frm_nms = list(self._frames['parameters'][pop_nm].keys())
        for par_nm in par_frm_nms: 
            if not par_nm in param_nms: 
                frm = self._frames['parameters'][pop_nm].pop(par_nm)
                frm.destroy()
                self._vars['parameters'][pop_nm].pop(par_nm)
        # place the new frames in the _frames dict 
        self._frames['parameters'][pop_nm] = new_par_frms
        #
        # SPECIES: 
        for spc_nm,frm in self._frames['species'][pop_nm].items(): frm.pack_forget() 
        self._frames['new_species'][pop_nm].pack_forget()
        new_spc_frms = OrderedDict()
        # save the relevant frames, destroy any that are obsolete
        spc_frm_nms = list(self._frames['species'][pop_nm].keys())
        for spc_nm in spc_frm_nms: 
            if spc_nm in self.sys.populations[pop_nm].basis:
                new_spc_frms[spc_nm] = self._frames['species'][pop_nm][spc_nm]
                # if the structure was changed to/from crystalline,
                # repack the specie frames 
                if (pop_struct == 'crystalline' and not 'coordx' in self._frames['specie_parameters'][pop_nm][spc_nm]) \
                or (not pop_struct == 'crystalline' and 'coordx' in self._frames['specie_parameters'][pop_nm][spc_nm]):
                    self._repack_specie_frame(pop_nm,spc_nm)
            else:
                frm = self._frames['species'][pop_nm].pop(spc_nm)
                frm.destroy()
                # TODO: remove refs to obsolete vars and widgets
                # that were children of this frame
        # check for new species, add frames as needed
        for spc_nm in self.sys.populations[pop_nm].basis.keys():
            if not spc_nm in new_spc_frms:
                new_spc_frms[spc_nm] = self._create_specie_frame(pop_nm,spc_nm)
        # place the new frames in the _frames dict 
        self._frames['species'][pop_nm] = new_spc_frms
        # PACKING 
        self._pack_setting_frames(pop_nm)
        self._pack_parameter_frames(pop_nm)
        self._pack_specie_frames(pop_nm)
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

    def _pack_specie_frames(self,pop_nm):
        n_stg_frms = len(self._frames['settings'][pop_nm])
        n_param_frms = len(self._frames['parameters'][pop_nm])
        n_specie_frms = len(self._frames['species'][pop_nm])
        for spc_idx, specief in enumerate(self._frames['species'][pop_nm].values()):
            specief.grid(row=1+n_stg_frms+n_param_frms+spc_idx,sticky='ew')
        bottom_row = 1+n_stg_frms+n_param_frms+n_specie_frms
        self._frames['new_species'][pop_nm].grid(row=bottom_row,sticky='ew')

    def _create_setting_frame(self,pop_nm,specie_nm,stg_nm):
        stg_vars = self._vars['settings'][pop_nm]
        stg_frames = self._frames['settings'][pop_nm]
        parent_obj = self.sys.populations[pop_nm]
        parent_frame = self._frames['populations'][pop_nm]
        if specie_nm:
            stg_vars = self._vars['specie_settings'][pop_nm][specie_nm]
            stg_frames = self._frames['specie_settings'][pop_nm][specie_nm]
            parent_obj = self.sys.populations[pop_nm].basis[specie_nm]
            parent_frame = self._frames['species'][pop_nm][specie_nm]
        stgf = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        stgf.grid_columnconfigure(1,weight=1)

        if xrsdefs.setting_datatypes[stg_nm] is str:
            stgv = tkinter.StringVar(parent_frame)
        elif xrsdefs.setting_datatypes[stg_nm] is int:
            stgv = tkinter.IntVar(parent_frame)
        elif xrsdefs.setting_datatypes[stg_nm] is float:
            stgv = tkinter.DoubleVar(parent_frame)
        stg_frames[stg_nm] = stgf
        stg_vars[stg_nm] = stgv

        stgl = tkinter.Label(stgf,text='{}:'.format(stg_nm),width=12,anchor='e')
        stgl.grid(row=0,column=0,sticky='e')
        s = xrsdefs.setting_defaults[stg_nm]
        if stg_nm in parent_obj.settings:
            s = parent_obj.settings[stg_nm]
        stgv.set(str(s))
        stge = self.connected_entry(stgf,stgv,
            partial(self._update_setting,pop_nm,specie_nm,stg_nm))
        stge.grid(row=0,column=1,sticky='ew')
        return stgf

    def _create_param_frame(self,pop_nm,specie_nm,param_nm):
        param_vars = self._vars['parameters'][pop_nm]
        param_frames = self._frames['parameters'][pop_nm]
        param_var_nm = pop_nm+'__'+param_nm
        if pop_nm == 'noise': 
            parent_frame = self._frames['noise_model']
            parent_obj = self.sys.noise_model
            param_default = xrsdefs.noise_param_defaults[param_nm]
        elif specie_nm:
            param_vars = self._vars['specie_parameters'][pop_nm][specie_nm]
            param_frames = self._frames['specie_parameters'][pop_nm][specie_nm]
            param_var_nm = pop_nm+'__'+specie_nm+'__'+param_nm
            parent_obj = self.sys.populations[pop_nm].basis[specie_nm]
            parent_frame = self._frames['species'][pop_nm][specie_nm]
            if param_nm in ['coordx','coordy','coordz']:
                param_default = xrsdefs.coord_default 
            else:
                param_default = xrsdefs.param_defaults[param_nm]
        else:
            parent_frame = self._frames['populations'][pop_nm]
            parent_obj = self.sys.populations[pop_nm]
            param_default = xrsdefs.param_defaults[param_nm]
        param_idx = len(param_frames)
        if not param_nm in param_vars: param_vars[param_nm] = {}

        paramf = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        paramf.grid_columnconfigure(2,weight=1)
        paramv = tkinter.DoubleVar(paramf)
        p = copy.deepcopy(param_default)
        if param_nm in ['coordx','coordy','coordz']:
            if param_nm == 'coordx': cidx = 0
            if param_nm == 'coordy': cidx = 1
            if param_nm == 'coordz': cidx = 2
            p.update(parent_obj.coordinates[cidx])
        elif param_nm in parent_obj.parameters:
            p.update(parent_obj.parameters[param_nm])
        param_frames[param_nm] = paramf
        param_vars[param_nm]['value'] = paramv

        pl = tkinter.Label(paramf,text='parameter:',anchor='e')
        pl.grid(row=0,column=0,sticky='e')
        pnml = tkinter.Label(paramf,text=param_nm,anchor='w') 
        pnml.grid(row=0,column=1,sticky='w')

        pfixvar = tkinter.BooleanVar(paramf)
        param_vars[param_nm]['fixed'] = pfixvar
        pfixvar.set(p['fixed'])
        psw = self.connected_checkbutton(paramf,pfixvar,
            partial(self._update_param,pop_nm,specie_nm,param_nm,'fixed'),'fixed')
        psw.grid(row=0,column=2,sticky='w')

        vl = tkinter.Label(paramf,text='value:',anchor='e')
        vl.grid(row=1,column=0,columnspan=1,sticky='e')
        paramv.set(p['value'])
        pe = self.connected_entry(paramf,paramv,
            partial(self._update_param,pop_nm,specie_nm,param_nm,'value'),16)
        pe.grid(row=1,column=1,columnspan=2,sticky='ew')

        pbndl = tkinter.Label(paramf,text='bounds:',anchor='e')
        pbndl.grid(row=2,column=0,sticky='e')
        lbndv = tkinter.StringVar(paramf)
        ubndv = tkinter.StringVar(paramf)
        param_vars[param_nm]['bounds']=[lbndv,ubndv]
        pbnde1 = self.connected_entry(paramf,lbndv,
            partial(self._update_param,pop_nm,specie_nm,param_nm,'bounds',0),8)
        lbndv.set(p['bounds'][0])
        ubndv.set(p['bounds'][1])
        pbnde2 = self.connected_entry(paramf,ubndv,
            partial(self._update_param,pop_nm,specie_nm,param_nm,'bounds',1),8)
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
        exprv.set(p['constraint_expr'])
        pexpe = self.connected_entry(paramf,exprv,
            partial(self._update_param,pop_nm,specie_nm,param_nm,'constraint_expr'),16)
        pexpe.grid(row=4,column=1,columnspan=2,sticky='ew')
        return paramf

    def _create_specie_frame(self,pop_nm,specie_nm):
        parent_frame = self._frames['populations'][pop_nm]
        specie = self.sys.populations[pop_nm].basis[specie_nm]
        specief = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        self._frames['species'][pop_nm][specie_nm] = specief
        pop_struct = self.sys.populations[pop_nm].structure
        #
        # NAME and FORM FACTOR: 
        speclf = tkinter.Frame(specief,bd=0)
        speclf.grid_columnconfigure(2,weight=1)
        specl = tkinter.Label(speclf,text='specie:',anchor='e')
        specnml = tkinter.Label(speclf,text=specie_nm,anchor='w')
        specl.grid(row=0,column=0,sticky='e')
        specnml.grid(row=0,column=1,padx=10,sticky='w')
        rmb = tkinter.Button(speclf,text='x',command=partial(self._remove_specie,pop_nm,specie_nm))
        rmb.grid(row=0,column=2,sticky='e')
        ffl = tkinter.Label(speclf,text='form factor:',width=12,anchor='e')
        ffl.grid(row=1,column=0,sticky='e')
        ffvar = tkinter.StringVar(speclf)
        ff_option_dict = OrderedDict.fromkeys(xrsdefs.form_factor_names)
        ffcb = tkinter.OptionMenu(speclf,ffvar,*ff_option_dict)
        ffvar.set(specie.form)
        ffvar.trace('w',partial(self._update_form_factor,pop_nm,specie_nm))
        ffcb.grid(row=1,column=1,sticky='ew') 
        self._vars['form_factors'][pop_nm][specie_nm] = ffvar
        speclf.grid(row=0,sticky='ew')
        #
        # SETTINGS:
        self._frames['specie_settings'][pop_nm][specie_nm] = OrderedDict()
        self._vars['specie_settings'][pop_nm][specie_nm] = OrderedDict()
        for istg,stg_nm in enumerate(xrsdefs.form_factor_settings[specie.form]):
            self._frames['specie_settings'][pop_nm][specie_nm][stg_nm] = \
            self._create_setting_frame(pop_nm,specie_nm,stg_nm)
        #
        # PARAMETERS:
        self._frames['specie_parameters'][pop_nm][specie_nm] = OrderedDict()
        self._vars['specie_parameters'][pop_nm][specie_nm] = OrderedDict()
        for param_nm in xrsdefs.form_factor_params[specie.form]:
            self._frames['specie_parameters'][pop_nm][specie_nm][param_nm] = \
            self._create_param_frame(pop_nm,specie_nm,param_nm)
        if pop_struct == 'crystalline': 
            self._frames['specie_parameters'][pop_nm][specie_nm]['coordx'] = \
            self._create_param_frame(pop_nm,specie_nm,'coordx')
            self._frames['specie_parameters'][pop_nm][specie_nm]['coordy'] = \
            self._create_param_frame(pop_nm,specie_nm,'coordy')
            self._frames['specie_parameters'][pop_nm][specie_nm]['coordz'] = \
            self._create_param_frame(pop_nm,specie_nm,'coordz')
        self._pack_specie_setting_frames(pop_nm,specie_nm)
        self._pack_specie_parameter_frames(pop_nm,specie_nm)
        return specief

    def _repack_specie_frame(self,pop_nm,specie_nm):
        pop_struct = self.sys.populations[pop_nm].structure
        spc_form = self.sys.populations[pop_nm].basis[specie_nm].form
        #
        # SETTINGS: 
        for stg_nm,frm in self._frames['specie_settings'][pop_nm][specie_nm].items(): frm.pack_forget() 
        new_stg_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        for stg_nm in xrsdefs.form_factor_settings[spc_form]:
            if stg_nm in self._frames['specie_settings'][pop_nm][specie_nm]:
                new_stg_frms[stg_nm] = self._frames['specie_settings'][pop_nm][specie_nm][stg_nm]
            else:
                new_stg_frms[stg_nm] = self._create_setting_frame(pop_nm,specie_nm,stg_nm)
        # destroy any frames that didn't get repacked
        stg_frm_nms = list(self._frames['specie_settings'][pop_nm][specie_nm].keys())
        for stg_nm in stg_frm_nms: 
            if not stg_nm in xrsdefs.form_factor_settings[spc_form]: 
                frm = self._frames['specie_settings'][pop_nm][specie_nm].pop(stg_nm)
                frm.destroy()
                self._vars['specie_settings'][pop_nm][specie_nm].pop(stg_nm)
        # place the new frames in the _frames dict 
        self._frames['specie_settings'][pop_nm][specie_nm] = new_stg_frms
        #
        # PARAMETERS: 
        for par_nm,frm in self._frames['specie_parameters'][pop_nm][specie_nm].items(): frm.pack_forget() 
        new_par_frms = OrderedDict()
        # save the relevant frames, create new ones as needed 
        param_nms = copy.deepcopy(xrsdefs.form_factor_params[spc_form])
        if pop_struct == 'crystalline': param_nms.extend(['coordx','coordy','coordz'])
        for par_nm in param_nms: 
            if par_nm in self._frames['specie_parameters'][pop_nm][specie_nm]:
                new_par_frms[par_nm] = self._frames['specie_parameters'][pop_nm][specie_nm][par_nm]
            else:
                new_par_frms[par_nm] = self._create_param_frame(pop_nm,specie_nm,par_nm)
        # destroy any frames that didn't get repacked
        par_frm_nms = list(self._frames['specie_parameters'][pop_nm][specie_nm].keys())
        for par_nm in par_frm_nms: 
            if not par_nm in param_nms: 
                frm = self._frames['specie_parameters'][pop_nm][specie_nm].pop(par_nm)
                frm.destroy()
                self._vars['specie_parameters'][pop_nm][specie_nm].pop(par_nm)
        # place the new frames in the _frames dict 
        self._frames['specie_parameters'][pop_nm][specie_nm] = new_par_frms
        # PACKING 
        self._pack_specie_setting_frames(pop_nm,specie_nm)
        self._pack_specie_parameter_frames(pop_nm,specie_nm)
        # update_idletasks() processes the frame changes, 
        # so that they are accounted for in control_canvas_configure()
        self.fit_gui.update_idletasks()
        self.control_canvas_configure()

    def _pack_specie_setting_frames(self,pop_nm,specie_nm):
        for stg_idx, stg_frm in enumerate(self._frames['specie_settings'][pop_nm][specie_nm].values()):
            stg_frm.grid(row=1+stg_idx,sticky='ew')

    def _pack_specie_parameter_frames(self,pop_nm,specie_nm):
        n_stg_frms = len(self._frames['specie_settings'][pop_nm][specie_nm])
        for param_idx, paramf in enumerate(self._frames['specie_parameters'][pop_nm][specie_nm].values()):
            paramf.grid(row=1+n_stg_frms+param_idx,sticky='ew')

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
        return npf

    def _create_new_specie_frame(self,pop_nm):
        pf = self._frames['populations'][pop_nm]
        nsf = tkinter.Frame(pf,bd=2,pady=10,padx=10,relief=tkinter.GROOVE)
        nsf.grid_columnconfigure(1,weight=1)
        self._frames['new_species'][pop_nm] = nsf
        self._vars['new_specie_names'][pop_nm] = tkinter.StringVar(pf)
        addl = tkinter.Label(nsf,text='new specie:',anchor='e')
        addl.grid(row=0,column=0,sticky='w')
        stnme = self.connected_entry(nsf,self._vars['new_specie_names'][pop_nm],None,12)
        stnme.grid(row=0,column=1,sticky='ew')
        stnme.bind('<Return>',partial(self._new_specie,pop_nm))
        addb = tkinter.Button(nsf,text='+',command=partial(self._new_specie,pop_nm))
        addb.grid(row=0,column=2,sticky='e')
        return nsf

    def _draw_plots(self):
        I_comp = draw_xrsd_fit(self.fig,self.sys,self.q,self.I,self.src_wl,self.dI,False)
        self.mpl_canvas.draw()
        self._update_fit_objective(I_comp)

    def _update_fit_objective(self,I_comp=None):
        obj_val = self.sys.evaluate_residual(
            self.q,self.I,self.src_wl,self.dI,
            self.error_weighted,self.logI_weighted,self.q_range,I_comp)
        self._vars['fit_control']['objective'].set(str(obj_val))

    def _update_param(self,pop_nm,specie_nm,param_nm,param_key,param_idx=None,event=None):
        # param_key should be 'value', 'fixed', 'bounds', or 'constraint_expr'
        # if param_key == 'bounds', param_idx must be 0 or 1
        vflag = self._validate_param(pop_nm,specie_nm,param_nm,param_key,param_idx)
        if vflag:
            if pop_nm == 'noise':
                x = self.sys.noise_model
                tkv = self._vars['parameters']['noise'][param_nm]
                xp = x.parameters[param_nm]
            elif specie_nm:
                x = self.sys.populations[pop_nm].basis[specie_nm] 
                tkv = self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]
                if param_nm in ['coordx','coordy','coordz']:
                    if param_nm == 'coordx': cidx = 0
                    if param_nm == 'coordy': cidx = 1
                    if param_nm == 'coordz': cidx = 2
                    xp = x.coordinates[cidx]
                else:
                    xp = x.parameters[param_nm]
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
                if param_nm in ['coordx','coordy','coordz']:
                    if param_nm == 'coordx': cidx = 0
                    if param_nm == 'coordy': cidx = 1
                    if param_nm == 'coordz': cidx = 2
                    x.update_coordinate(cidx,new_param)
                else:
                    x.update_parameter(param_nm,new_param)
                self._draw_plots()
        return vflag

    def _update_setting(self,pop_nm,specie_nm,stg_nm,event=None):
        vflag = self._validate_setting(pop_nm,specie_nm,stg_nm) 
        if vflag:
            if specie_nm: 
                x = self.sys.populations[pop_nm].basis[specie_nm] 
                tkv = self._vars['specie_settings'][pop_nm][specie_nm][stg_nm]
            else:
                x = self.sys.populations[pop_nm]
                tkv = self._vars['settings'][pop_nm][stg_nm]
            new_val = tkv.get()
            if not new_val == x.settings[stg_nm]:
                x.update_setting(stg_nm,new_val)
                if specie_nm:
                    self._repack_specie_frame(pop_nm,specie_nm)
                else:
                    self._repack_pop_frame(pop_nm)
                self._draw_plots()
        return vflag

    def _validate_setting(self,pop_nm,specie_nm,stg_nm):
        """Validate a setting Var entry and set its value in self.sys"""
        if specie_nm:
            old_stg = self.sys.populations[pop_nm].basis[specie_nm].settings[stg_nm]
            tkv = self._vars['specie_settings'][pop_nm][specie_nm][stg_nm]
        else:
            x = self.sys.populations[pop_nm].settings[stg_nm]
            tkv = self._vars['settings'][pop_nm][stg_nm]
        is_valid = True
        try:
            new_val = tkv.get()
        except:
            is_valid = False
            tkv.set(x)
        return is_valid

    def _validate_param(self,pop_nm,specie_nm,param_nm,param_key,param_idx=None):
        """Validate a parameter Var entry and set its value in self.sys 

        If the entry is valid, the Variable is set to the Entry's value. 
        If the entry is not valid, the Variable is reset.
        
        Parameters
        ----------
        pop_nm : string 
        specie_nm : string 
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
        elif specie_nm:
            if param_nm == 'coordx': 
                x = self.sys.populations[pop_nm].basis[specie_nm].coordinates[0]
            elif param_nm == 'coordy': 
                x = self.sys.populations[pop_nm].basis[specie_nm].coordinates[1]
            elif param_nm == 'coordz': 
                x = self.sys.populations[pop_nm].basis[specie_nm].coordinates[2]
            else:
                x = self.sys.populations[pop_nm].basis[specie_nm].parameters[param_nm]
            tkvs = self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]
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
            self.sys.add_population(new_nm,'diffuse')
            self._frames['new_population'].pack_forget() 
            self._frames['populations'][new_nm] = self._create_pop_frame(new_nm)
            npops = len(self._frames['populations'])
            self._frames['populations'][new_nm].grid(row=1+npops,padx=2,pady=2,sticky='ew') 
            self._frames['new_population'].grid(row=2+npops,padx=2,pady=2,sticky='ew') 
            # update_idletasks() processes the new frame,
            # so that it is accounted for in control_canvas_configure()
            self.fit_gui.update_idletasks()
            self.control_canvas_configure()

    def _new_specie(self,pop_nm,event=None):
        specie_nm = self._vars['new_specie_names'][pop_nm].get()
        if specie_nm and not specie_nm in self.sys.populations[pop_nm].basis:
            self.sys.populations[pop_nm].add_specie(specie_nm,'atomic')
            self._frames['new_species'][pop_nm].pack_forget() 
            self._frames['species'][pop_nm][specie_nm] = self._create_specie_frame(pop_nm,specie_nm)
            n_stg_frms = len(self._frames['settings'][pop_nm])
            n_param_frms = len(self._frames['parameters'][pop_nm])
            n_specie_frms = len(self._frames['species'][pop_nm])
            self._frames['species'][pop_nm][specie_nm].grid(row=1+n_stg_frms+n_param_frms+n_specie_frms,sticky='ew')  
            self._frames['new_species'][pop_nm].grid(row=2+n_stg_frms+n_param_frms+n_specie_frms,sticky='ew')
            # update_idletasks() processes the new frame,
            # so that it is accounted for in control_canvas_configure()
            self.fit_gui.update_idletasks()
            self.control_canvas_configure()

    def _update_structure(self,pop_nm,*event_args):
        s = self._vars['structures'][pop_nm].get()
        if not s == self.sys.populations[pop_nm].structure:
            try:
                self.sys.populations[pop_nm].set_structure(s)
            except StructureFormException:
                self._vars['structures'][pop_nm].set(self.sys.populations[pop_nm].structure)
            except:
                raise
            self._repack_pop_frame(pop_nm)
            self._draw_plots()

    def _update_form_factor(self,pop_nm,specie_nm,*event_args):
        f = self._vars['form_factors'][pop_nm][specie_nm].get()
        if not f == self.sys.populations[pop_nm].basis[specie_nm].form:
            try:
                self.sys.populations[pop_nm].set_form(specie_nm,f)
            except StructureFormException:
                self._vars['form_factors'][pop_nm][specie_nm].set(
                self.sys.populations[pop_nm].basis[specie_nm].form)
            except:
                raise
            self._repack_specie_frame(pop_nm,specie_nm)
            self._draw_plots()

    def _remove_population(self,pop_nm):
        self.sys.remove_population(pop_nm)
        self._repack_pop_frames()
        self._draw_plots()

    def _remove_specie(self,pop_nm,specie_nm):
        self.sys.populations[pop_nm].remove_specie(specie_nm)
        self._repack_pop_frame(pop_nm)
        self._draw_plots()

    def _fit(self):
        sys_opt = xrsdsys.fit(
            self.sys,
            self.q,self.I,self.src_wl,self.dI,
            self.error_weighted,self.logI_weighted,self.q_range
            )
        self.sys.update_from_dict(sys_opt.to_dict())
        self._update_parameters() 
        self._draw_plots()

    def _update_parameters(self):
        for param_nm,par in self.sys.noise_model.parameters.items():
            self._vars['parameters']['noise'][param_nm]['value'].set(par['value'])
        for pop_nm,pop in self.sys.populations.items():        
            for param_nm in pop.parameters.keys():
                self._vars['parameters'][pop_nm][param_nm]['value'].set(
                pop.parameters[param_nm]['value'])
            for specie_nm in pop.basis.keys():
                for param_nm in pop.basis[specie_nm].parameters.keys():
                    self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]['value'].set(
                        pop.basis[specie_nm].parameters[param_nm]['value'])
                    if pop.structure == 'crystalline':
                        for ic, coord_tag in enumerate(['coordx','coordy','coordz']):
                            self._vars['specie_parameters'][pop_nm][specie_nm][coord_tag]['value'].set(
                                pop.basis[specie_nm].coordinates[ic]['value'])

    def _estimate(self):
        # TODO
        # sys_est = xrsdmods.estimate(
        #    self.sys,
        #    self.q_I[:,0],self.q_I[:,1]
        #    )
        # update self.sys
        # repack everything 
        # update fit objective
        # self.draw_plots()
        pass

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
    def connected_entry(parent,tkvar,cbfun=None,entry_width=20):
        if cbfun:
            # piggyback on entry validation to update internal data
            # NOTE: validatecommand must return a boolean, or else it will disconnect quietly
            e = tkinter.Entry(parent,width=entry_width,textvariable=tkvar,validate="focusout",validatecommand=cbfun)
            # also respond to the return key
            e.bind('<Return>',cbfun)
        else:
            e = tkinter.Entry(parent,width=entry_width,textvariable=tkvar)
        return e

    @staticmethod
    def connected_checkbutton(parent,boolvar,cbfun=None,label=''):
        if cbfun:
            e = tkinter.Checkbutton(parent,text=label,variable=boolvar,command=cbfun,anchor='w')
        else:
            e = tkinter.Checkbutton(parent,text=label,variable=boolvar,anchor='w')
        return e

