from collections import OrderedDict
from functools import partial
import copy
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
mplv = matplotlib.__version__
mplvmaj = int(mplv.split('.')[0])
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
if mplvmaj > 2:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
else:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg

from .. import *
from . import plot_xrsd_fit, draw_xrsd_fit
from .. import system as xrsdsys

if sys.version_info[0] < 3:
    import Tkinter as tkinter
else:
    import tkinter

def run_fit_gui(system,source_wavelength,q,I,dI=None,
    error_weighted=True,
    logI_weighted=True,
    q_range=[0.,float('inf')],
    good_fit_prior=False):
    gui = XRSDFitGUI(system,source_wavelength,q,I,dI,error_weighted,logI_weighted,q_range,good_fit_prior)
    sys_opt, good_fit = gui.start()
    # collect results and return
    return sys_opt, good_fit

# TODO: when a param is fixed or has a constraint set,
#   make the entry widget read-only

# TODO: make plot frame zoom-able (add matplotlib toolbar?)

# TODO: whenever any param or selection is updated,
#   ensure other params/selections remain valid,
#   wrt constraints as well as wrt supported options.
#   NOTE: This logic should be in xrsdkit.system.*

# TODO: add basic IO

# TODO: when frames are updated, make repacking optional,
#   and only do so when necessary, because repacking is slow

class XRSDFitGUI(object):

    def __init__(self,system,source_wavelength,
        q,I,dI=None,
        error_weighted=True,
        logI_weighted=True,
        q_range=[0.,float('inf')],
        good_fit_prior=False):

        super(XRSDFitGUI, self).__init__()
        self.q = q
        self.I = I
        self.dI = dI
        self.src_wl = source_wavelength
        self.sys = system
        self.sys_opt = system
        self.error_weighted = error_weighted
        self.logI_weighted = logI_weighted
        self.q_range = q_range
        self.good_fit = good_fit_prior

        self.fit_gui = tkinter.Tk()
        self.fit_gui.protocol('WM_DELETE_WINDOW',self._cleanup)
        # setup the main gui objects
        self._build_gui()
        # create the plots
        self._build_plot_widgets()
        # create the widgets for control 
        self._build_control_widgets()
        # draw the plots...
        self._draw_plots()
        self.fit_gui.geometry('1100x700')

    def start(self):
        # start the tk loop
        self.fit_gui.mainloop()
        # after the loop, return the (optimized) system
        return self.sys_opt, self.good_fit

    def _build_gui(self):
        self.fit_gui.title('xrsd profile fitter')
        # a horizontal scrollbar and a main canvas belong to the main gui:
        scrollbar = tkinter.Scrollbar(self.fit_gui,orient='horizontal')
        fit_gui_canvas = tkinter.Canvas(self.fit_gui)
        scrollbar.pack(side=tkinter.BOTTOM,fill=tkinter.X)
        fit_gui_canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)
        scrollbar.config(command=fit_gui_canvas.xview)
        fit_gui_canvas.config(xscrollcommand=scrollbar.set)
        # the main widget will be a frame,
        # displayed as a window item on the main canvas:
        self.main_frame = tkinter.Frame(fit_gui_canvas,bd=4,relief=tkinter.SUNKEN)
        main_frame_window = fit_gui_canvas.create_window(0,0,window=self.main_frame,anchor='nw')
        # _canvas_configure() ensures that the window item and scrollbar
        # remain the correct size for the underlying widget
        fit_gui_canvas.bind("<Configure>",partial(self._canvas_configure,fit_gui_canvas,self.main_frame,main_frame_window))

    @staticmethod
    def _canvas_configure(canvas,widget,window,event):
        # Resize the frame to match the canvas.
        # The window is the "canvas item" that displays the widget.
        minw = widget.winfo_reqwidth()
        minh = widget.winfo_reqheight()
        if canvas.winfo_width() >= minw:
            minw = canvas.winfo_width()
        if canvas.winfo_height() >= minh:
            minh = canvas.winfo_height()
        canvas.itemconfigure(window,width=minw,height=minh)
        canvas.config(scrollregion=canvas.bbox(tkinter.ALL))

    def _cleanup(self):
        # remove references to all gui objects, widgets, etc. 
        #self._reset_control_widgets() 
        self.fit_gui.quit()
        self.fit_gui.destroy()

    def _build_plot_widgets(self):
        # the main frame contains a plot frame on the left,
        # containing a canvas, which contains a window item,
        # which displays a view on a plot widget 
        # built from FigureCanvasTkAgg.get_tk_widget()
        plot_frame = tkinter.Frame(self.main_frame,bd=4,relief=tkinter.SUNKEN)
        plot_frame.pack(side=tkinter.LEFT,fill=tkinter.BOTH,expand=True,padx=2,pady=2)
        self.fig = plot_xrsd_fit(self.sys,self.src_wl,self.q,self.I,self.dI,False)
        plot_frame_canvas = tkinter.Canvas(plot_frame)
        yscr = tkinter.Scrollbar(plot_frame)
        yscr.pack(side=tkinter.RIGHT,fill='y')
        plot_frame_canvas.pack(fill='both',expand=True)
        plot_frame_canvas.config(yscrollcommand=yscr.set)
        yscr.config(command=plot_frame_canvas.yview)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig,plot_frame_canvas)
        self.plot_canvas = self.mpl_canvas.get_tk_widget()
        plot_canvas_window = plot_frame_canvas.create_window(0,0,window=self.plot_canvas,anchor='nw')
        plot_frame_canvas.bind("<Configure>",partial(
            self._canvas_configure,plot_frame_canvas,self.plot_canvas,plot_canvas_window))
        self.mpl_canvas.draw()

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
        # TODO: figure out a way to set or control the width of the control widget
        self.control_widget = tkinter.Frame(control_frame_canvas)
        control_canvas_window = control_frame_canvas.create_window((0,0),window=self.control_widget,anchor='nw')
        control_frame_canvas.bind("<Configure>",partial(
            self._canvas_configure,control_frame_canvas,self.control_widget,control_canvas_window))
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
            self._create_pop_frame(pop_nm)
        self._create_new_pop_frame()

    def _create_fit_control_frame(self):
        # TODO: callbacks for fit controls to update instance attributes
        # TODO: wavelength entry
        cf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        cf.grid_columnconfigure(1,weight=1)
        cf.grid_columnconfigure(2,weight=1)
        self._frames['fit_control'] = cf
        self._vars['fit_control']['objective'] = tkinter.StringVar(cf)
        self._vars['fit_control']['error_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['logI_weighted'] = tkinter.BooleanVar(cf)
        self._vars['fit_control']['q_range'] = [tkinter.DoubleVar(cf),tkinter.DoubleVar(cf)]
        
        objl = tkinter.Label(cf,text='objective:',anchor='e')
        objl.grid(row=0,column=0,sticky='e')
        rese = tkinter.Entry(cf,width=10,state='readonly',textvariable=self._vars['fit_control']['objective'])
        rese.grid(row=0,column=1,sticky='ew')
        self._vars['fit_control']['good_fit'] = tkinter.BooleanVar(cf)
        fitcb = tkinter.Checkbutton(cf,text='Good fit', variable=self._vars['fit_control']['good_fit'])
        fitcb.grid(row=0,column=2,sticky='ew')
        self._vars['fit_control']['good_fit'].set(self.good_fit)

        q_range_lbl = tkinter.Label(cf,text='q-range:',anchor='e')
        q_range_lbl.grid(row=1,column=0,sticky='e')
        q_lo_ent = tkinter.Entry(cf,width=8,textvariable=self._vars['fit_control']['q_range'][0])
        q_hi_ent = tkinter.Entry(cf,width=8,textvariable=self._vars['fit_control']['q_range'][1])
        q_lo_ent.grid(row=1,column=1,sticky='ew')
        q_hi_ent.grid(row=1,column=2,sticky='ew')
        self._vars['fit_control']['q_range'][0].set(self.q_range[0])
        self._vars['fit_control']['q_range'][1].set(self.q_range[1])

        ewtcb = tkinter.Checkbutton(cf,text='error weighted',variable=self._vars['fit_control']['error_weighted'])
        ewtcb.select()
        ewtcb.grid(row=2,column=0,sticky='w')
        logwtbox = tkinter.Checkbutton(cf,text='log(I) weighted',variable=self._vars['fit_control']['logI_weighted'])
        logwtbox.select()
        logwtbox.grid(row=3,column=0,sticky='w')

        estbtn = tkinter.Button(cf,text='Estimate',width=8,command=self._estimate)
        estbtn.grid(row=2,column=1,rowspan=2,sticky='nesw')
        fitbtn = tkinter.Button(cf,text='Fit',width=8,command=self._fit)
        fitbtn.grid(row=2,column=2,rowspan=2,sticky='nesw')
        cf.pack(pady=2,padx=2,fill='both',expand=True)

    def _update_fit_objective(self):
        errwtd = self._vars['fit_control']['error_weighted'].get()
        logwtd = self._vars['fit_control']['logI_weighted'].get()
        qrng = [self._vars['fit_control']['q_range'][0].get(),\
            self._vars['fit_control']['q_range'][1].get()]
        obj_val = self.sys.evaluate_residual(self.src_wl,self.q,self.I,self.dI,errwtd,logwtd,qrng)
        self._vars['fit_control']['objective'].set(str(obj_val))

    def _create_noise_frame(self):
        nf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        self._frames['noise_model'] = nf
        nmf = tkinter.Frame(nf,bd=0) 
        nl = tkinter.Label(nmf,text='noise model:',width=12,anchor='e',padx=10)
        nl.pack(side=tkinter.LEFT)
        ntpvar = tkinter.StringVar(nmf)
        ntp_option_dict = list(noise_model_names)
        ntpcb = tkinter.OptionMenu(nmf,ntpvar,*ntp_option_dict)
        ntpvar.set(self.sys.noise_model.model)
        ntpvar.trace('w',self._update_noise_frame)
        ntpcb.pack(side=tkinter.LEFT,fill='x')
        self._vars['noise_model'] = ntpvar
        nmf.pack(fill='x',expand=True)

        self._frames['parameters']['noise'] = OrderedDict()
        self._vars['parameters']['noise'] = OrderedDict()
        for ip,noise_param_nm in enumerate(noise_params[self.sys.noise_model.model]):
            self._create_param_widget('noise',None,noise_param_nm) 
        nf.pack(pady=2,padx=2,fill='x',expand=True)
        
    def _create_pop_frame(self,pop_nm):
        pop = self.sys.populations[pop_nm]
        pf = tkinter.Frame(self.control_widget,bd=4,pady=10,padx=10,relief=tkinter.RAISED)
        self._frames['populations'][pop_nm] = pf
        pop_struct = self.sys.populations[pop_nm].structure

        # sub-frame for name and structure 
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
        str_option_dict = OrderedDict.fromkeys(structure_names)
        strcb = tkinter.OptionMenu(plf,strvar,*str_option_dict)
        strvar.set(pop_struct)
        strvar.trace('w',partial(self._update_structure,pop_nm))
        strcb.grid(row=1,column=1,sticky='ew')
        self._vars['structures'][pop_nm] = strvar
        plf.pack(fill='x',expand=True)

        self._frames['settings'][pop_nm] = OrderedDict()
        self._vars['settings'][pop_nm] = OrderedDict()
        for stg_nm in structure_settings[pop_struct]:
            self._create_setting_widget(pop_nm,None,stg_nm)

        self._frames['parameters'][pop_nm] = OrderedDict()
        self._vars['parameters'][pop_nm] = OrderedDict()
        param_nms = copy.deepcopy(structure_params[pop_struct])
        if pop_struct == 'crystalline': 
            param_nms.extend(copy.deepcopy(crystalline_structure_params[pop_struct]))
        if pop_struct == 'disordered': 
            param_nms.extend(copy.deepcopy(disordered_structure_params[pop_struct]))
        for param_nm in param_nms:
            self._create_param_widget(pop_nm,None,param_nm)

        self._frames['species'][pop_nm] = OrderedDict()
        self._frames['specie_parameters'][pop_nm] = OrderedDict()
        self._frames['specie_settings'][pop_nm] = OrderedDict()
        self._vars['form_factors'][pop_nm] = OrderedDict()
        self._vars['specie_parameters'][pop_nm] = OrderedDict()
        self._vars['specie_settings'][pop_nm] = OrderedDict()
        for specie_nm,specie in self.sys.populations[pop_nm].basis.items():
            self._create_specie_frame(pop_nm,specie_nm)
        self._create_new_specie_frame(pop_nm)

        pf.pack(pady=2,padx=2,fill='x',expand=True)

    def _create_specie_frame(self,pop_nm,specie_nm):
        parent_frame = self._frames['populations'][pop_nm]
        specie = self.sys.populations[pop_nm].basis[specie_nm]
        specief = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        self._frames['species'][pop_nm][specie_nm] = specief
        pop_struct = self.sys.populations[pop_nm].structure

        # sub-frame for name and form factor
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
        ff_option_dict = OrderedDict.fromkeys(form_factor_names)
        ffcb = tkinter.OptionMenu(speclf,ffvar,*ff_option_dict)
        ffvar.set(specie.form)
        ffvar.trace('w',partial(self._update_form_factor))
        ffcb.grid(row=1,column=1,sticky='ew') 
        self._vars['form_factors'][pop_nm][specie_nm] = ffvar
        speclf.pack(fill='x',expand=True)

        self._frames['specie_settings'][pop_nm][specie_nm] = OrderedDict()
        self._vars['specie_settings'][pop_nm][specie_nm] = OrderedDict()
        for istg,stg_nm in enumerate(form_factor_settings[specie.form]):
            self._create_setting_widget(pop_nm,specie_nm,stg_nm)

        self._frames['specie_parameters'][pop_nm][specie_nm] = OrderedDict()
        self._vars['specie_parameters'][pop_nm][specie_nm] = OrderedDict()
        for ip,param_nm in enumerate(form_factor_params[specie.form]):
            self._create_param_widget(pop_nm,specie_nm,param_nm)

        specief.pack(fill='x',expand=True)

        if pop_struct == 'crystalline': 
            self._create_coordinate_widgets(pop_nm,specie_nm)

    def _create_coordinate_widgets(self,pop_nm,specie_nm):
        self._create_param_widget(pop_nm,specie_nm,'coordx')
        self._create_param_widget(pop_nm,specie_nm,'coordy')
        self._create_param_widget(pop_nm,specie_nm,'coordz')

    def _create_param_widget(self,pop_nm,specie_nm,param_nm):
        param_vars = self._vars['parameters'][pop_nm]
        param_frames = self._frames['parameters'][pop_nm]
        param_var_nm = pop_nm+'__'+param_nm
        if pop_nm == 'noise': 
            parent_frame = self._frames['noise_model']
            parent_obj = self.sys.noise_model
            param_default = noise_param_defaults[param_nm]
        elif specie_nm:
            param_vars = self._vars['specie_parameters'][pop_nm][specie_nm]
            param_frames = self._frames['specie_parameters'][pop_nm][specie_nm]
            param_var_nm = pop_nm+'__'+specie_nm+'__'+param_nm
            parent_obj = self.sys.populations[pop_nm].basis[specie_nm]
            parent_frame = self._frames['species'][pop_nm][specie_nm]
            if param_nm in ['coordx','coordy','coordz']:
                param_default = coord_default 
            else:
                param_default = param_defaults[param_nm]
        else:
            parent_frame = self._frames['populations'][pop_nm]
            parent_obj = self.sys.populations[pop_nm]
            param_default = param_defaults[param_nm]
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

        paramf.pack(fill='x',expand=True)

    def _create_setting_widget(self,pop_nm,specie_nm,stg_nm):
        stg_vars = self._vars['settings'][pop_nm]
        stg_frames = self._frames['settings'][pop_nm]
        parent_obj = self.sys.populations[pop_nm]
        parent_frame = self._frames['populations'][pop_nm]
        stg_default = setting_defaults[stg_nm]
        if specie_nm:
            stg_vars = self._vars['specie_settings'][pop_nm][specie_nm]
            stg_frames = self._frames['specie_settings'][pop_nm][specie_nm]
            parent_obj = self.sys.populations[pop_nm].basis[specie_nm]
            parent_frame = self._frames['species'][pop_nm][specie_nm]

        stgf = tkinter.Frame(parent_frame,bd=2,pady=4,padx=10,relief=tkinter.GROOVE)
        stgf.grid_columnconfigure(1,weight=1)

        if setting_datatypes[stg_nm] is str:
            stgv = tkinter.StringVar(parent_frame)
        elif setting_datatypes[stg_nm] is int:
            stgv = tkinter.IntVar(parent_frame)
        elif setting_datatypes[stg_nm] is float:
            stgv = tkinter.DoubleVar(parent_frame)
        stg_frames[stg_nm] = stgf
        stg_vars[stg_nm] = stgv

        stgl = tkinter.Label(stgf,text='{}:'.format(stg_nm),width=12,anchor='e')
        stgl.grid(row=0,column=0,sticky='e')
        s = setting_defaults[stg_nm]
        if stg_nm in parent_obj.settings:
            s = parent_obj.settings[stg_nm]
        stgv.set(str(s))
        stge = self.connected_entry(stgf,stgv,
            partial(self._update_setting,pop_nm,specie_nm,stg_nm))
        stge.grid(row=0,column=1,sticky='ew')
        stgf.pack(fill='x',expand=True)

    def _draw_plots(self):
        draw_xrsd_fit(self.fig,self.sys_opt,self.src_wl,self.q,self.I,None,False)
        self.mpl_canvas.draw()
        self._update_fit_objective()

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
        npf.pack(pady=2,padx=2,fill='x',expand=True)

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
        nsf.pack(fill='x',expand=True)

    def _update_param(self,pop_nm,specie_nm,param_nm,param_key,param_idx=None,event=None):
        # param_key should be 'value', 'fixed', 'bounds', or 'constraint_expr'
        # if param_key == 'bounds', param_idx must be 0 or 1
        vflag = self._validate_param(pop_nm,specie_nm,param_nm,param_key,param_idx)
        #print('{}.{}.{}.{}.{}: {}'.format(pop_nm,specie_nm,param_nm,param_key,param_idx,vflag))
        if vflag:
            if pop_nm == 'noise':
                x = self.sys.noise_model
                tkv = self._vars['parameters']['noise'][param_nm]
            elif specie_nm:
                x = self.sys.populations[pop_nm].basis[specie_nm] 
                tkv = self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]
            else: 
                x = self.sys.populations[pop_nm]
                tkv = self._vars['parameters'][pop_nm][param_nm]
            if param_nm in ['coordx','coordy','coordz']:
                if param_nm == 'coordx': cidx = 0
                if param_nm == 'coordy': cidx = 1
                if param_nm == 'coordz': cidx = 2
                xp = parent_obj.coordinates[cidx]
            else:
                xp = x.parameters[param_nm]
            new_param = copy.deepcopy(xp)
            if param_idx in [0,1]: 
                new_val = tkv[param_key][param_idx].get()
                if new_val in ['None','none','']:
                    new_param[param_key][param_idx] = None
                else: 
                    new_param[param_key][param_idx] = float(new_val) 
            else:
                new_val = tkv[param_key].get()
                if param_key == 'constraint_expr':
                    # TODO: further validate constraint expr.?
                    if new_val in ['None','none','']:
                        new_val = None
                new_param[param_key] = new_val 
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
        print('{}.{}.{}: {}'.format(pop_nm,specie_nm,stg_nm,vflag))
        if vflag:
            x = self.sys.populations[pop_nm]
            tkv = self._vars['settings'][pop_nm]
            if specie_nm: 
                x = x.basis[specie_nm] 
                tkv = self._vars['specie_settings'][pop_nm][specie_nm][stg_nm]
            new_val = tkv.get()
            x.update_setting(stg_nm,new_val)
            self._update_specie_frame(pop_nm,specie_nm) 
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
                # TODO: if param_key == 'constraint_expr', validate the expression
            except:
                is_valid = False
                tkvs[param_key].set(old_val)
        return is_valid

    def _update_structure(self,pop_nm,event=None):
        s = self._vars['structures'][pop_nm].get()
        if not s == self.sys.populations[pop_nm].structure:
            self.sys.populations[pop_nm].set_structure(s)
            self._update_pop_frame(pop_nm)

    def _update_form_factor(self,pop_nm,specie_nm,event=None):
        f = self._vars['form_factors'][pop_nm][specie_nm].get()
        if not f == self.sys.populations[pop_nm].basis[specie_nm].form:
            self.sys.populations[pop_nm].basis[specie_nm].set_form(f)
            self._update_specie_frame(pop_nm,specie_nm)

    def _remove_population(self,pop_nm):
        # remove the population from self.sys
        self.sys.remove_population(pop_nm)
        # remove any associated frames and vars
        self._update_control_frame()
        self._draw_plots()

    def _remove_specie(self,pop_nm,specie_nm):
        # remove the specie from the population
        self.sys.populations[pop_nm].remove_specie(specie_nm)
        # remove any associated frames and vars
        self._update_pop_frame(pop_nm) 
        self._draw_plots()

    def _update_control_frame(self):
        popfrm_nms = list(self._frames['populations'].keys())
        for pnm in popfrm_nms:
            if not pnm in self.sys.populations:
                self._frames['settings'].pop(pnm)
                self._frames['parameters'].pop(pnm)
                self._frames['species'].pop(pnm)
                self._frames['specie_settings'].pop(pnm)
                self._frames['specie_parameters'].pop(pnm)
                self._frames['new_species'].pop(pnm)
                self._frames['populations'][pnm].pack_forget() 
                self._frames['populations'][pnm].destroy()
                self._frames['populations'].pop(pnm)
                self._vars['structures'].pop(pnm)
                self._vars['settings'].pop(pnm)
                self._vars['parameters'].pop(pnm)
                self._vars['form_factors'].pop(pnm)
                self._vars['specie_settings'].pop(pnm)
                self._vars['specie_parameters'].pop(pnm)
                self._vars['new_specie_names'].pop(pnm)
        self._update_noise_frame()
        for pnm in self.sys.populations.keys():
            if pnm in self._frames['populations']:
                self._update_pop_frame(pnm)
            else:
                self._frames['new_population'].pack_forget() 
                self._create_pop_frame(pnm)
                self._frames['new_population'].pack(pady=2,padx=2,fill='x',expand=True)

    def _update_noise_frame(self):
        parfrm_nms = list(self._frames['parameters']['noise'].keys())
        # remove any obsolete param widgets
        for par_nm in parfrm_nms:
            if not par_nm in self.sys.noise_model.parameters:
                self._frames['parameters']['noise'][par_nm].pack_forget()
                self._frames['parameters']['noise'][par_nm].destroy()
                self._frames['parameters']['noise'].pop(par_nm)
                self._vars['parameters']['noise'].pop(par_nm)
        # repack param widgets, add any that are missing
        param_nms = copy.deepcopy(noise_params[self.sys.noise_model.model])
        for param_nm in param_nms:
            if not param_nm in self._frames['parameters']['noise']:
                self._create_param_widget('noise',None,param_nm)
            else:
                self._frames['parameters']['noise'][param_nm].pack_forget()
                self._frames['parameters']['noise'][param_nm].pack(fill='x',expand=True)
                self._vars['parameters']['noise'][param_nm]['value'].set(
                self.sys.noise_model.parameters[param_nm]['value'])

    def _update_pop_frame(self,pop_nm):
        stgfrm_nms = list(self._frames['settings'][pop_nm].keys())
        parfrm_nms = list(self._frames['parameters'][pop_nm].keys())
        spcfrm_nms = list(self._frames['species'][pop_nm].keys())
        pop_struct = self.sys.populations[pop_nm].structure
        # remove any obsolete setting widgets
        for stg_nm in stgfrm_nms:
            if not stg_nm in self.sys.populations[pop_nm].settings:
                self._frames['settings'][pop_nm][stg_nm].pack_forget()
                self._frames['settings'][pop_nm][stg_nm].destroy()
                self._frames['settings'][pop_nm].pop(stg_nm)
                self._vars['settings'][pop_nm].pop(stg_nm)
        # remove any obsolete param widgets
        for par_nm in parfrm_nms:
            if not par_nm in self.sys.populations[pop_nm].parameters:
                self._frames['parameters'][pop_nm][par_nm].pack_forget()
                self._frames['parameters'][pop_nm][par_nm].destroy()
                self._frames['parameters'][pop_nm].pop(par_nm)
                self._vars['parameters'][pop_nm].pop(par_nm)
        # remove any obsolete specie widgets
        for spc_nm in spcfrm_nms:
            if not spc_nm in self.sys.populations[pop_nm].basis:
                self._frames['specie_parameters'][pop_nm].pop(spc_nm)
                self._frames['specie_settings'][pop_nm].pop(spc_nm)
                self._frames['species'][pop_nm][spc_nm].pack_forget()
                self._frames['species'][pop_nm][spc_nm].destroy()
                self._frames['species'][pop_nm].pop(spc_nm)
                self._vars['form_factors'][pop_nm].pop(spc_nm)
                self._vars['specie_settings'][pop_nm].pop(spc_nm)
                self._vars['specie_parameters'][pop_nm].pop(spc_nm)
        # repack setting widgets, add any that are missing
        for stg_nm in structure_settings[pop_struct]:
            if not stg_nm in self._frames['settings'][pop_nm]:
                self._create_setting_widget(pop_nm,None,stg_nm)
            else:
                self._frames['settings'][pop_nm][stg_nm].pack_forget()
                self._frames['settings'][pop_nm][stg_nm].pack(fill='x',expand=True)
        # repack param widgets, add any that are missing
        param_nms = copy.deepcopy(structure_params[pop_struct])
        if pop_struct == 'crystalline': 
            pop_lat = self.sys.populations[pop_nm].settings['lattice']
            param_nms.extend(copy.deepcopy(crystalline_structure_params[pop_lat]))
        if pop_struct == 'disordered': 
            pop_int = self.sys.populations[pop_nm].settings['interaction']
            param_nms.extend(copy.deepcopy(disordered_structure_params[pop_int]))
        for param_nm in param_nms:
            if not param_nm in self._frames['parameters'][pop_nm]:
                self._create_param_widget(pop_nm,None,param_nm)
            else:
                self._frames['parameters'][pop_nm][param_nm].pack_forget()
                self._frames['parameters'][pop_nm][param_nm].pack(fill='x',expand=True)
                self._vars['parameters'][pop_nm][param_nm]['value'].set(
                self.sys.populations[pop_nm].parameters[param_nm]['value'])
        # update and repack any specie widgets, add any that are missing
        self._frames['new_species'][pop_nm].pack_forget()
        for specie_nm in self.sys.populations[pop_nm].basis.keys():
            if not specie_nm in self._frames['species'][pop_nm]:
                self._create_specie_frame(pop_nm,specie_nm)
            else:
                self._update_specie_frame(pop_nm,specie_nm)
                self._frames['species'][pop_nm][specie_nm].pack_forget()
                self._frames['species'][pop_nm][specie_nm].pack(fill='x',expand=True)
        self._frames['new_species'][pop_nm].pack(fill='x',expand=True)

    def _update_specie_frame(self,pop_nm,specie_nm):
        stgfrm_nms = list(self._frames['specie_settings'][pop_nm][specie_nm].keys())
        parfrm_nms = list(self._frames['specie_parameters'][pop_nm][specie_nm].keys())
        pop_struct = self.sys.populations[pop_nm].structure
        specie_form = self.sys.populations[pop_nm].basis[specie_nm].form
        # remove any obsolete setting widgets
        for stg_nm in stgfrm_nms:
            if not stg_nm in self.sys.populations[pop_nm].basis[specie_nm].settings:
                self._frames['specie_settings'][pop_nm][specie_nm][stg_nm].pack_forget()
                self._frames['specie_settings'][pop_nm][specie_nm][stg_nm].destroy()
                self._frames['specie_settings'][pop_nm][specie_nm].pop(stg_nm)
                self._vars['specie_settings'][pop_nm][specie_nm].pop(stg_nm)
        # remove any obsolete param widgets
        for par_nm in parfrm_nms:
            if (not par_nm in self.sys.populations[pop_nm].basis[specie_nm].parameters 
                and not par_nm in ['coordx','coordy','coordz']) \
            or (par_nm in ['coordx','coordy','coordz'] 
                and not pop_struct == 'crystalline'):
                self._frames['specie_parameters'][pop_nm][specie_nm][par_nm].pack_forget()
                self._frames['specie_parameters'][pop_nm][specie_nm][par_nm].destroy()
                self._frames['specie_parameters'][pop_nm][specie_nm].pop(par_nm)
                self._vars['specie_parameters'][pop_nm][specie_nm].pop(par_nm)
        # repack setting widgets, add any that are missing
        for stg_nm in form_factor_settings[specie_form]:
            if not stg_nm in self._frames['specie_settings'][pop_nm][specie_nm]:
                self._create_setting_widget(pop_nm,specie_nm,stg_nm)
            else:
                self._frames['specie_settings'][pop_nm][specie_nm][stg_nm].pack_forget()
                self._frames['specie_settings'][pop_nm][specie_nm][stg_nm].pack(fill='x',expand=True)
        # repack param widgets, add any that are missing
        param_nms = copy.deepcopy(form_factor_params[specie_form])
        for param_nm in param_nms:
            if not param_nm in self._frames['specie_parameters'][pop_nm][specie_nm]:
                self._create_param_widget(pop_nm,specie_nm,param_nm)
            else:
                self._frames['specie_parameters'][pop_nm][specie_nm][param_nm].pack_forget()
                self._frames['specie_parameters'][pop_nm][specie_nm][param_nm].pack(fill='x',expand=True)
                if param_nm in ['coordx','coordy','coordz']:
                    if param_nm == 'coordx': cidx = 0
                    if param_nm == 'coordy': cidx = 1
                    if param_nm == 'coordz': cidx = 2
                    self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]['value'].set(
                    self.sys.populations[pop_nm].basis[specie_nm].coordinates[cidx]['value'])
                else:
                    self._vars['specie_parameters'][pop_nm][specie_nm][param_nm]['value'].set(
                    self.sys.populations[pop_nm].basis[specie_nm].parameters[param_nm]['value'])
        # if the structure has been updated to 'crystalline',
        # but coordinate widgets do not exist,
        # create them now
        if pop_struct == 'crystalline' \
        and not 'coordx' in self._frames['specie_parameters'][pop_nm][specie_nm].keys():
            self._create_coordinate_widgets(pop_nm,specie_nm)

    def _new_population(self,event=None):
        new_nm = self._vars['new_population_name'].get()
        if new_nm and not new_nm in self.sys.populations:
            self.sys.add_population(new_nm,'diffuse')
            self._frames['new_population'].pack_forget() 
            self._create_pop_frame(new_nm)
            self._frames['new_population'].pack(pady=2,padx=2,fill='x',expand=True)

    def _new_specie(self,pop_nm,event=None):
        specie_nm = self._vars['new_specie_names'][pop_nm].get()
        if specie_nm and not specie_nm in self.sys.populations[pop_nm].basis:
            self.sys.populations[pop_nm].add_specie(specie_nm,'atomic')
            self._frames['new_species'][pop_nm].pack_forget() 
            self._create_specie_frame(pop_nm,specie_nm)
            self._frames['new_species'][pop_nm].pack(fill='x',expand=True)

    def _fit(self):
        sys_opt = xrsdsys.fit(
            self.sys,self.src_wl,
            self.q,self.I,self.dI,
            self.error_weighted,self.logI_weighted,self.q_range
            )
        self.sys.update_from_dict(sys_opt.to_dict())
        self._update_control_frame() 
        self._update_fit_objective
        self._draw_plots()

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

