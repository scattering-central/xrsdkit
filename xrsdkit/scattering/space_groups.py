from collections import OrderedDict
import copy

crystal_systems = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
lattice_systems = ['triclinic','monoclinic','orthorhombic','tetragonal','rhombohedral','hexagonal','cubic']  

# enumerate all the space groups for each crystal system
crystal_space_groups = dict(
    triclinic = {
        1:'P1',2:'P-1'
        },
    monoclinic = {
        3:'P2',4:'P2(1)',5:'C2',6:'Pm',7:'Pc',8:'Cm',9:'Cc',10:'P2/m',
        11:'P2(1)/m',12:'C2/m',13:'P2/c',14:'P2(1)/c',15:'C2/c'
        },
    orthorhombic = {
        16:'P222',17:'P222(1)',18:'P2(1)2(1)2',19:'P2(1)2(1)2(1)',20:'C222(1)',
        21:'C222',22:'F222',23:'I222',24:'I2(1)2(1)2(1)',25:'Pmm2',26:'Pmc2(1)',
        27:'Pcc2',28:'Pma2',29:'Pca2(1)',30:'Pnc2',31:'Pmn2(1)',32:'Pba2',
        33:'Pna2(1)',34:'Pnn2',35:'Cmm2',36:'Cmc2(1)',37:'Ccc2',38:'Amm2',
        39:'Abm2',40:'Ama2',41:'Aba2',42:'Fmm2',43:'Fdd2',44:'Imm2',45:'Iba2',
        46:'Ima2',47:'Pmmm',48:'Pnnn',49:'Pccm',50:'Pban',51:'Pmma',52:'Pnna',
        53:'Pmna',54:'Pcca',55:'Pbam',56:'Pccn',57:'Pbcm',58:'Pnnm',59:'Pmmn',
        60:'Pbcn',61:'Pbca',62:'Pnma',63:'Cmcm',64:'Cmca',65:'Cmmm',66:'Cccm',
        67:'Cmma',68:'Ccca',69:'Fmmm',70:'Fddd',71:'Immm',72:'Ibam',73:'Ibca',
        74:'Imma'
        },
    tetragonal = {
        75:'P4',76:'P4(1)',77:'P4(2)',78:'P4(3)',79:'I4',80:'I4(1)',81:'P-4',
        82:'I-4',83:'P4/m',84:'P4(2)/m',85:'P4/n',86:'P4(2)/n',87:'I4/m',
        88:'I4(1)/a',89:'P422',90:'P42(1)2',91:'P4(1)22',92:'P4(1)2(1)2',
        93:'P4(2)22',94:'P4(2)2(1)2',95:'P4(3)22',96:'P4(3)2(1)2',97:'I422',
        98:'I4(1)22',99:'P4mm',100:'P4bm',101:'P4(2)cm',102:'P4(2)nm',
        103:'P4cc',104:'P4nc',105:'P4(2)mc',106:'P4(2)bc',107:'I4mm',108:'I4cm',
        109:'I4(1)md',110:'I4(1)cd',111:'P-42m',112:'P-42c',113:'P-42(1)m',
        114:'P-42(1)c',115:'P-4m2',116:'P-4c2',117:'P-4b2',118:'P-4n2',
        119:'I-4m2',120:'I-4c2',121:'I-42m',122:'I-42d',123:'P4/mmm',
        124:'P4/mcc',125:'P4/nbm',126:'P4/nnc',127:'P4/mbm',128:'P4/mnc',
        129:'P4/nmm',130:'P4/ncc',131:'P4(2)/mmc',132:'P4(2)/mcm',
        133:'P4(2)/nbc',134:'P4(2)/nnm',135:'P4(2)/mbc',136:'P4(2)/mnm',
        137:'P4(2)/nmc',138:'P4(2)/ncm',139:'I4/mmm',140:'I4/mcm',
        141:'I4(1)/amd',142:'I4(1)/acd'
        },
    trigonal = {
        143:'P3',144:'P3(1)',145:'P3(2)',146:'R3',147:'P-3',148:'R-3',
        149:'P312',150:'P321',151:'P3(1)12',152:'P3(1)21',153:'P3(2)12',
        154:'P3(2)21',155:'R32',156:'P3m1',157:'P31m',158:'P3c1',159:'P31c',
        160:'R3m',161:'R3c',162:'P-31m',163:'P-31c',164:'P-3m1',165:'P-3c1',
        166:'R-3m',167:'R-3c'
        },
    hexagonal = {
        168:'P6',169:'P6(1)',170:'P6(5)',171:'P6(2)',172:'P6(4)',173:'P6(3)',
        174:'P-6',175:'P6/m',176:'P6(3)/m',177:'P622',178:'P6(1)22',
        179:'P6(5)22',180:'P6(2)22',181:'P6(4)22',182:'P6(3)22',183:'P6mm',
        184:'P6cc',185:'P6(3)cm',186:'P6(3)mc',187:'P-6m2',188:'P-6c2',
        189:'P-62m',190:'P-62c',191:'P6/mmm',192:'P6/mcc',193:'P6(3)/mcm',
        194:'P6(3)/mmc'
        },
    cubic = {
        195:'P23',196:'F23',197:'I23',198:'P2(1)3',199:'I2(1)3',200:'Pm-3',
        201:'Pn-3',202:'Fm-3',203:'Fd-3',204:'Im-3',205:'Pa-3',206:'Ia-3',
        207:'P432',208:'P4(2)32',209:'F432',210:'F4(1)32',211:'I432',
        212:'P4(3)32',213:'P4(1)32',214:'I4(1)32',215:'P-43m',216:'F4-3m',
        217:'I-43m',218:'P-43n',219:'F-43c',220:'I-43d',221:'Pm-3m',222:'Pn-3n',
        223:'Pm-3n',224:'Pn-3m',225:'Fm-3m',226:'Fm-3c',227:'Fd-3m',228:'Fd-3c',
        229:'Im-3m',230:'Ia-3d'
        }
    )

all_space_groups = [] 
for xsys in crystal_systems:
    all_space_groups.extend(crystal_space_groups[xsys].values())

# enumerate the space groups for each lattice and centering:
lattice_space_groups = dict.fromkeys(lattice_systems) 
lattice_space_groups['triclinic'] = dict(P = crystal_space_groups['triclinic'])
lattice_space_groups['monoclinic'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['monoclinic'].items() if sgv[0]=='P']),
    C = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['monoclinic'].items() if sgv[0]=='C'])
    )
lattice_space_groups['orthorhombic'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['orthorhombic'].items() if sgv[0]=='P']),
    C = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['orthorhombic'].items() if sgv[0]=='C']),
    I = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['orthorhombic'].items() if sgv[0]=='I']),
    F = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['orthorhombic'].items() if sgv[0]=='F'])
    )
lattice_space_groups['tetragonal'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['tetragonal'].items() if sgv[0]=='P']),
    I = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['tetragonal'].items() if sgv[0]=='I'])
    )
lattice_space_groups['rhombohedral'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['trigonal'].items() if sgv[0]=='R']),
    )
lattice_space_groups['hexagonal'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['trigonal'].items() if sgv[0]=='P']),
    HCP = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['trigonal'].items() if sgv[0]=='P'])
    )
lattice_space_groups['hexagonal']['P'].update(crystal_space_groups['hexagonal'])
lattice_space_groups['hexagonal']['HCP'].update(crystal_space_groups['hexagonal'])
lattice_space_groups['cubic'] = dict(
    P = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['cubic'].items() if sgv[0]=='P']),
    I = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['cubic'].items() if sgv[0]=='I']),
    F = dict([(sgk,sgv) for sgk,sgv in crystal_space_groups['cubic'].items() if sgv[0]=='F'])
    )

# default space groups for each bravais lattice 
default_space_groups = dict(
    cubic = {'P':'Pm-3m','I':'Im-3m','F':'Fm-3m'},
    hexagonal = {'P':'P6/mmm','HCP':'P6(3)/mmc'},
    rhombohedral = {'P':'R-3m'},
    tetragonal = {'P':'P4/mmm','I':'I4/mmm'},
    orthorhombic = {'P':'Pmmm','I':'Immm','F':'Fmmm','C':'Cmmm'},
    monoclinic = {'P':'P2/m','C':'C2/m'},
    triclinic = {'P':'P-1'}
    )

# point groups for each crystal class
crystal_point_groups = dict(
    triclinic = ['1','-1'],
    monoclinic = ['2','2/m','222','m','mm2','mmm'],
    orthorhombic = ['2','2/m','222','m','mm2','mmm'],
    tetragonal = ['4','-4','4/m','422','4mm','-42m','4/mmm'],
    trigonal = ['3','-3','32','3m','-3m'],
    hexagonal = ['6','-6','6/m','622','6mm','-6m2','6/mmm'],
    cubic = ['23','m-3','432','-43m','m-3m']  
    )
all_point_groups = []
all_point_groups.extend(crystal_point_groups['triclinic'])
all_point_groups.extend(crystal_point_groups['monoclinic'])
all_point_groups.extend(crystal_point_groups['orthorhombic'])
all_point_groups.extend(crystal_point_groups['tetragonal'])
all_point_groups.extend(crystal_point_groups['trigonal'])
all_point_groups.extend(crystal_point_groups['hexagonal'])
all_point_groups.extend(crystal_point_groups['cubic'])

# map each space group to its underlying point group
sg_point_groups = OrderedDict()
sg_point_groups[crystal_space_groups['triclinic'][1]] = '1' 
sg_point_groups[crystal_space_groups['triclinic'][2]] = '-1' 
for isg in range(3,6): sg_point_groups[crystal_space_groups['monoclinic'][isg]] = '2'
for isg in range(6,10): sg_point_groups[crystal_space_groups['monoclinic'][isg]] = 'm'
for isg in range(10,16): sg_point_groups[crystal_space_groups['monoclinic'][isg]] = '2/m'
for isg in range(16,25): sg_point_groups[crystal_space_groups['orthorhombic'][isg]] = '222'
for isg in range(25,47): sg_point_groups[crystal_space_groups['orthorhombic'][isg]] = 'mm2'
for isg in range(47,75): sg_point_groups[crystal_space_groups['orthorhombic'][isg]] = 'mmm'
for isg in range(75,81): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '4'
sg_point_groups[crystal_space_groups['tetragonal'][81]] = '-4'
sg_point_groups[crystal_space_groups['tetragonal'][82]] = '-4'
for isg in range(83,89): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '4/m'
for isg in range(89,99): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '422'
for isg in range(99,111): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '4mm'
for isg in range(111,123): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '-42m'
for isg in range(123,143): sg_point_groups[crystal_space_groups['tetragonal'][isg]] = '4/mmm'
for isg in range(143,147): sg_point_groups[crystal_space_groups['trigonal'][isg]] = '3'
sg_point_groups[crystal_space_groups['trigonal'][147]] = '-3'
sg_point_groups[crystal_space_groups['trigonal'][148]] = '-3'
for isg in range(149,156): sg_point_groups[crystal_space_groups['trigonal'][isg]] = '32'
for isg in range(156,162): sg_point_groups[crystal_space_groups['trigonal'][isg]] = '3m'
for isg in range(162,168): sg_point_groups[crystal_space_groups['trigonal'][isg]] = '-3m'
for isg in range(168,174): sg_point_groups[crystal_space_groups['hexagonal'][isg]] = '6'
sg_point_groups[crystal_space_groups['hexagonal'][174]] = '-6'
sg_point_groups[crystal_space_groups['hexagonal'][175]] = '6/m'
sg_point_groups[crystal_space_groups['hexagonal'][176]] = '6/m'
for isg in range(177,183): sg_point_groups[crystal_space_groups['hexagonal'][isg]] = '622'
for isg in range(183,187): sg_point_groups[crystal_space_groups['hexagonal'][isg]] = '6mm'
for isg in range(187,191): sg_point_groups[crystal_space_groups['hexagonal'][isg]] = '-6m2'
for isg in range(191,195): sg_point_groups[crystal_space_groups['hexagonal'][isg]] = '6/mmm'
for isg in range(195,200): sg_point_groups[crystal_space_groups['cubic'][isg]] = '23'
for isg in range(200,207): sg_point_groups[crystal_space_groups['cubic'][isg]] = 'm-3'
for isg in range(207,215): sg_point_groups[crystal_space_groups['cubic'][isg]] = '432'
for isg in range(215,221): sg_point_groups[crystal_space_groups['cubic'][isg]] = '-43m'
for isg in range(221,231): sg_point_groups[crystal_space_groups['cubic'][isg]] = 'm-3m'



