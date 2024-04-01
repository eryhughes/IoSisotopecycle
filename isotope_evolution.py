### isotope_evolution.py ###

### import python packages ###
import pandas as pd
import numpy as np
import math
import datetime

###############
### OPTIONS ###
###############

# options for calculating fractionation factors
nofrac = "no" # if "yes", all fractionation factors = 1 (i.e., no fractionation); if "no", fractionation factors are pre-determined values
MAF = "no" # if "yes", mass-anomalous (i.e., with powers not equal to canonical values) is possible; if "no", fractionation factors follow canonical power laws

# options for calculating cap-delta values
# "O2017_eq5" = equation (5) in Ono (2017): D3XS = d3XS - e*d34S 
# "O2017_eq6" = equation (6) in Ono (2017):  D3X = d3X - ((d34+1.)^e3X - 1.)
# "FW2003_pg3" = equations on pg. 3 of Farquar & Wing (2003): D3XS = d3XS - 1000*((1+d34S/1000)^e - 1)
MIF_eq = "FW2003_pg3" 

#####################
### SYSTEM VALUES ###
#####################

# reservoir mass bounds
M_mass_i_min = 2.95E+21 # mantle S mol
M_mass_i_max = 2.24E+22
M_mass_i_min_sil = 7.26E+22 # mantle silicate kg
M_mass_i_max_sil = 7.63E+22 # kg
F_mass_i_min = 0. # frost S mol (crust)
F_mass_i_max = 1.92367E+20
SS_mass_i_min = 0. # silicates/sulfates S mol (crust)
SS_mass_i_max = 7.9E+19

# canonical mass-dependent fractionation powers
e33 = 0.515
e36 = 1.9

# mass-dependent fractionation factors
def a_3X(a34,X):
    if X == 33:
        e = e33
    elif X == 36:
        e = e36
    a = a34**e
    return a

# calculate fractionation factors
def alphas(process,X,nofrac,MAF):
    if nofrac == "yes":
        a = 1.
    else:
        if process == "gr": # gravity
            a34 = 0.916820406
        elif process == "th": # thermal
            a34 = 0.9998
        elif process == "ed": # electro-dissociation
            a34 = 0.994
        elif process == "pd": # photo-dissocation
            a34 = 1.008725
        elif process == "hg": # homogeneous gas equilibria
            a34 = 0.998
        elif process == "sq": # sulfate sequestration
            a34 = 0.9985
        elif process == "cf": # core formation
            a34 = 0.9998
        elif process in ["pu","ac","ei","rc","ec","pi","mm","dg","xt","fr","rm","vp","dm"]: # pick-up, assymetric charge exchange, electro-ionisation, resonant charge exchange, elastic collision, photo ionisation, mantle melting, degassing, crystallisation, frost remobilisation, return to mantle, vapor pressure isotope effect, deep mantle mixing
            a34 = 1. 
        if X == 34:
            a = a34
        elif X in [33,36]:
            if MAF == "no":
                a33 = a_3X(a34,33)
                a36 = a_3X(a34,36)
            elif MAF == "yes":
                if process == "gr": # gravity
                    a33 = 0.957445318
                    a36 = 0.840434662
                elif process == "ed": # electro-dissociation
                    a33 = 0.996
                    a36 = 0.989
                elif process == "pd": # photo-dissocation
                    a33 = 1.008225
                    a36 = 1.0068
                elif process in ["th","pu","ac","ei","rc","ec","pi","mm","dg","xt","fr","hg","rm","vp","sq","dm","cf"]: # thermal, pick-up, assymetric charge exchange, electro-ionisation, resonant charge exchange, elastic collision, photo ionisation, mantle melting, degassing, crystallisation, frost remobilisation, homogeneous gas equilibria, return to mantle, vapor pressure isotope effect, sulfate sequestration, deep mantle mixing, core formation
                    a33 = a_3X(a34,33)
                    a36 = a_3X(a34,36)
            if X == 33:
                a = a33
            elif X == 36:
                a = a36
    return a
                  
# alpha 34
a34_gr = alphas("gr",34,nofrac,MAF) # gravity
a34_th = alphas("th",34,nofrac,MAF) # thermal
a34_pu = alphas("pu",34,nofrac,MAF) # pick-up
a34_ed = alphas("ed",34,nofrac,MAF) # 0.994 # electrodissociation
a34_ac = alphas("ac",34,nofrac,MAF) # assymetric charge exchange
a34_ei = alphas("ei",34,nofrac,MAF) # electro ionisation
a34_rc = alphas("rc",34,nofrac,MAF) # resonant charge exchange
a34_ec = alphas("ec",34,nofrac,MAF) # elastic collision
a34_pi = alphas("pi",34,nofrac,MAF) # photo ionisation
a34_mm = alphas("mm",34,nofrac,MAF) # mantle melting
a34_dg = alphas("dg",34,nofrac,MAF) # degassing
a34_xt = alphas("xt",34,nofrac,MAF) # crystallisation
a34_fr = alphas("fr",34,nofrac,MAF) # remobilisation
a34_pd = alphas("pd",34,nofrac,MAF) # photodissocation
a34_hg = alphas("hg",34,nofrac,MAF) # homogeneous gas equilibria
a34_rm = alphas("rm",34,nofrac,MAF) # return to mantle
a34_vp = alphas("vp",34,nofrac,MAF) # vapor pressure isotope effect
a34_sq = alphas("sq",34,nofrac,MAF) # sulfate sequestration
a34_dm = alphas("dm",34,nofrac,MAF) # deep mantle mixing
a34_mo = a34_mm*a34_dg
a34_pl = a34_mm*a34_xt
a34_cf = alphas("cf",34,nofrac,MAF) # core formation

# alpha 33
a33_gr = alphas("gr",33,nofrac,MAF) # gravity
a33_th = alphas("th",33,nofrac,MAF) # thermal
a33_pu = alphas("pu",33,nofrac,MAF) # pick-up
a33_ed = alphas("ed",33,nofrac,MAF) # electrodissociation
a33_ac = alphas("ac",33,nofrac,MAF) # assymetric charge exchange
a33_ei = alphas("ei",33,nofrac,MAF) # electro ionisation
a33_rc = alphas("rc",33,nofrac,MAF) # resonant charge exchange
a33_ec = alphas("ec",33,nofrac,MAF) # elastic collision
a33_pi = alphas("pi",33,nofrac,MAF) # photo ionisation
a33_mm = alphas("mm",33,nofrac,MAF) # mantle melting
a33_dg = alphas("dg",33,nofrac,MAF) # degassing
a33_xt = alphas("xt",33,nofrac,MAF) # crystallisation
a33_fr = alphas("fr",33,nofrac,MAF) # rembolisation
a33_pd = alphas("pd",33,nofrac,MAF) # photodissocation
a33_hg = alphas("hg",33,nofrac,MAF) # homogeneous gas equilibria
a33_rm = alphas("rm",33,nofrac,MAF) # return to mantle
a33_vp = alphas("vp",33,nofrac,MAF) # vapor pressure isotope effect
a33_sq = alphas("sq",33,nofrac,MAF) # sulfate sequestration
a33_dm = alphas("dm",33,nofrac,MAF) # deep mantle mixing
a33_mo = a33_mm*a33_dg
a33_pl = a33_mm*a33_xt
a33_cf = alphas("cf",33,nofrac,MAF) # core formation
    
# alpha 36
a36_gr = alphas("gr",36,nofrac,MAF) # gravity
a36_th = alphas("th",36,nofrac,MAF) # thermal
a36_pu = alphas("pu",36,nofrac,MAF) # pick-up
a36_ed = alphas("ed",36,nofrac,MAF) # electrodissociation
a36_ac = alphas("ac",36,nofrac,MAF) # assymetric charge exchange
a36_ei = alphas("ei",36,nofrac,MAF) # electro ionisation
a36_rc = alphas("rc",36,nofrac,MAF) # resonant charge exchange
a36_ec = alphas("ec",36,nofrac,MAF) # elastic collision
a36_pi = alphas("pi",36,nofrac,MAF) # photo ionisation
a36_mm = alphas("mm",36,nofrac,MAF) # mantle melting
a36_dg = alphas("dg",36,nofrac,MAF) # degassing
a36_xt = alphas("xt",36,nofrac,MAF) # crystallisation
a36_fr = alphas("fr",36,nofrac,MAF) # frost remobolisation
a36_pd = alphas("pd",36,nofrac,MAF) # photodissocation
a36_hg = alphas("hg",36,nofrac,MAF) # homogeneous gas equilibria 
a36_rm = alphas("rm",36,nofrac,MAF) # return to mantle
a36_vp = alphas("vp",36,nofrac,MAF) # vapor pressure isotope effect
a36_sq = alphas("sq",36,nofrac,MAF) # sulfur sequestration
a36_dm = alphas("dm",36,nofrac,MAF) # deep mantle mixing
a36_mo = a36_mm*a36_dg
a36_pl = a36_mm*a36_xt
a36_cf = alphas("cf",36,nofrac,MAF) # core formation

# rate SO2 kg/s
rate_kgs = {"pd_min":3028.,"pd_max":3533.,
                  "pi_min":205.,"pi_max":326.,
                  "ei_min":105.,"ei_max":105.,
                  "ed_min":1500.,"ed_max":1500.,
                  "ac_min":30.,"ac_max":30.,
                  "rc_min":1600.,"rc_max":1600.,
                  "ec_min":110.,"ec_max":110.,
                  "pu_min":1000.}

Io_SA = 4.174e13 # Io's surface area in m2
sil_mag_rho = 3000. # silicate melt density in kg/m3

# composition f Vienna Canyon Diablo Troilite
VCDT = {"34":1./22.6436, #Ding+2001
       "33":1./126.948, #Ding+2001
       "36":1./6515.} #Ding+2001

###################
### CONVERSIONS ###
###################

def R2d(R,n,VCDT): # R to delta value
    if n == 34:
        std = VCDT["34"]
    elif n == 33:
        std = VCDT["33"]
    elif n == 36:
        std = VCDT["36"]
    result = ((R - std)/std)*1000.
    return result

def d2R(d,n,VCDT): # delta value to R
    if n == 34:
        std = VCDT["34"]
    elif n == 33:
        std = VCDT["33"]
    elif n == 36:
        std = VCDT["36"]
    result = ((d/1000.)*std)+std
    return result

def ROT(a,c,b,d): # from ratio (e.g., 33S/32S = a) to over total (e.g., 33S/ST = X), where c, b, d are the other ratios (i.e., 32S/32S, 34S/32S, 36S/32S)
    X = a/(a+b+c+d)
    return X

def kgs2mSs(kgs): # kg/s SO2 to moles S/s
    molSs = (kgs*1000.)/64.
    return molSs

def MIF(R33,R34,R36): # cap-delta notation
    d33 = R2d(R33,33,VCDT)
    d34 = R2d(R34,34,VCDT)
    d36 = R2d(R36,36,VCDT)
    
    def calc_MIF(d34,d3X,e3X):
        if MIF_eq == "O2017_eq5":
            D3X = d3X - e3X*d34 # used in first submission - eq. 5 Ono (2017)
        elif MIF_eq == "O2017_eq6":
            D3X = d3X - ((d34+1.)**e3X - 1.) # eq. 6 Ono (2017)
        elif MIF_eq == "FW2003_pg3":
            D3X = d3X - 1000.*((1. + d34/1000.)**e3X - 1.) # Farquar & Wing (2003)
        return D3X

    D33 = calc_MIF(d34,d33,e33)
    D36 = calc_MIF(d34,d36,e36)
    return D33, D36

def calc_rate(f,r_min,r_max): # rate kg/s SO2 - plasma and photodissociation 
    if f == "N":
        rate = 0.
    elif f <= 1.:
        rate = (f*(r_max-r_min))+r_min
    elif f > 1.:
        rate = f*((r_max+r_min)/2.)
    return rate

#################
### FUNCTIONS ###
#################

# initial mass
def initial_mass (minimum,maximum,fraction): 
    total = minimum + fraction*(maximum - minimum)
    return total

# total across reservoirs
def reservoir_totals(M,DM,F,SS,S): 
    total = M + DM + F + SS + S
    return total

# mass balance as a fraction of initial
def mass_balance(initial,now):
    mb = (initial - now)/initial
    return mb

# sulfur concentration from total sulfur in moles and total mass of reservoir in kg
def S_conc(ST,mass):
    conc = ((ST*32.)/1000.)/mass
    return conc

# mole S/s from mantle melting
def molSs_mm(mag_kgs,mag_S,f): 
    mm = ((f*mag_kgs*mag_S)*1000.)/32.
    return mm

# reservoir composition
def reservoir_R_S(S32, S33, S34, S36):
    if S32 == 0.:
        return "","","",0.,0.,0.,0.,0.
    else:
        R33 = S33/S32 # 33S/32S
        R34 = S34/S32 # 34S/32S
        R36 = S36/S32 # 36S/32S
        ST = S32 + S33 + S34 + S36
        T32 = S32/ST # 32S/ST
        T33 = S33/ST # 33S/ST
        T34 = S34/ST # 34S/ST
        T36 = S36/ST # 36S/ST
        return R33, R34, R36, T32, T33, T34, T36, ST

# delta and cap-delta composition for reservoir    
def reservoir_isotope(S32, S33, S34, S36):
    if S32 == 0.:
        return "","","","","","","","",0.,"",
    else:
        R33, R34, R36, T32, T33, T34, T36, TS = reservoir_R_S(S32, S33, S34, S36)
        log_ST = math.log10(TS)
        d33, d34, d36 = R2d(R33,33,VCDT), R2d(R34,34,VCDT), R2d(R36,36,VCDT)
        D33, D36 = MIF(R33,R34,R36)
        return R33, R34, R36, d33, d34, d36, D33, D36, TS, log_ST
   
# calculate initial isotopic composition of reservoir
def initial_reservoir(ST,d33,d34,d36):
    if ST > 0.:
       log_ST = math.log10(ST)
       R33, R34, R36 = d2R(d33,33,VCDT), d2R(d34,34,VCDT), d2R(d36,36,VCDT)
       T32, T33, T34, T36 = ROT(1.,R33,R34,R36), ROT(R33,R34,R36,1.), ROT(R34,R36,1.,R33), ROT(R36,1.,R33,R34)
       D33, D36 = MIF(R33,R34,R36)
       S32, S33, S34, S36 = ST*T32, ST*T33, ST*T34, ST*T36
       return log_ST, R33, R34, R36, T32, T33, T34, T36, D33, D36, S32, S33, S34, S36
    else:
        return "","","","","","","","","","",0.,0.,0.,0.
    
# composition due to a process - mo, rf, sl, sq, hg
def process_R_S(F,res_33R,res_34R,res_36R,a33,a34,a36):
    if F == 0.:
        return "","","",0.,0.,0.,0.
    else:
        R33 = res_33R*a33
        R34 = res_34R*a34
        R36 = res_36R*a36
        S32 = F*ROT(1.,R33,R34,R36)
        S33 = F*ROT(R33,R34,R36,1.)
        S34 = F*ROT(R34,R36,1.,R33)
        S36 = F*ROT(R36,1.,R33,R34)
        return R33, R34, R36, S32, S33, S34, S36

# mass balance
def mantle(M,mo,pl,rs,dm): # mantle
    M_ = M - mo - pl + rs + dm # mantle outgassing, plutons, return silicate+sulfate, deep mantle mantle
    return M_
def deep_mantle(DM,dm): # mantle
    DM_ = DM - dm # deep mantle mixing
    return DM_
def frost(F,fr,bu,hg): # crustal frost
    F_ = F - fr + bu + hg # frost remobilisation, burial
    return F_
def silsulf(SS,rs,sq,pl): # crustal frost
    SS_ = SS - rs + sq + pl # return silicate+sulfate, sequestration, pluton
    return SS_
def space(S,pu): # space
    S_ = S + pu # space loss
    return S_

# combination alphas
def a_bu(sn_F, pd_F, a_vp, a_pd): # burial
    bu_F = sn_F + pd_F
    a = a_vp*(sn_F/bu_F) + a_pd*(pd_F/bu_F)
    return a
def a_pr(alphas,rates): # plasma reactions
    alpha = alphas["pu"]*(alphas["gr"]*alphas["th"]*(alphas["ed"]*(rates["ed"]/rates["pr"]) + alphas["ac"]*(rates["ac"]/rates["pr"]) + alphas["ei"]*(rates["ei"]/rates["pr"]) + alphas["rc"]*(rates["rc"]/rates["pr"]) + alphas["ec"]*(rates["ec"]/rates["pr"])) + alphas["pi"]*(rates["pi"]/rates["pr"]))
    return alpha
    
# newton raphson solver
def newton_raphson(x0,constants,e1,step,eqs,deriv,maxiter=1000):

    def dx(x,eqs):
        f_ = eqs(x,constants)
        result =(abs(0-f_))
        return result
    
    def nr(x0,eqs,deriv):
        f_ = eqs(x0,constants)
        df_ = deriv
        x0 = x0 - step*(f_/df_)
        return x0
    
    # create results table        
    results = pd.DataFrame([["guessx","diff","step","f","df","f/df"]])  
    
    delta1 = dx(x0,eqs)
    n=1
    results1 = pd.DataFrame([[x0,delta1,step]]) 
    results = pd.concat([results, results1], ignore_index=True)
    
    while delta1 > e1:
        f_ = eqs(x0,constants)
        df_ = deriv(x0,constants)
        x0 = x0 - step*(f_/df_)
        n=n+1.
        #while x0 < 0.:
        #    step = step/10.
        #    x0 = x0 - step*(f_/df_)
        delta1 = dx(x0,eqs)      
        results1 = pd.DataFrame([[x0,delta1,step,f_,df_,f_/df_]])
        results = pd.concat([results, results1], ignore_index=True)  
        if n > 10:
            results.to_csv('results_nr.csv', index=False, header=False) 
    return x0    

# partition material between space and burial
def fractionate(constants,nr_step,nr_tol,guessx):     
    
    def crust_comp(S32C, constants):
        STT, STE, STC, S32T, S33T, S34T, S36T, a33, a34, a36 = constants
        S33C = (S33T*S32C)/(S32C + S32T*a33 - S32C*a33)
        S34C = (S34T*S32C)/(S32C + S32T*a34 - S32C*a34)
        S36C = (S36T*S32C)/(S32C + S32T*a36 - S32C*a36)        
        return S33C, S34C, S36C
        
    def f(S32C, constants):
        STT, STE, STC, S32T, S33T, S34T, S36T, a33, a34, a36 = constants
        S33C, S34C, S36C = crust_comp(S32C, constants)
        mb = STC - S32C - S33C - S34C - S36C
        return mb
    
    def df(S32C, constants):
        STT, STE, STC, S32T, S33T, S34T, S36T, a33, a34, a36 = constants
        diff = -S32C*S33T*(a33 - 1.)/(-S32C*a33 + S32C + S32T*a33)**2. - S32C*S34T*(a34 - 1.)/(-S32C*a34 + S32C + S32T*a34)**2. - S32C*S36T*(a36 - 1.)/(-S32C*a36 + S32C + S32T*a36)**2. - S33T/(-S32C*a33 + S32C + S32T*a33) - S34T/(-S32C*a34 + S32C + S32T*a34) - S36T/(-S32C*a36 + S32C + S32T*a36) - 1.
        return diff
    
    def reservoirs(S32C, constants):
        STT, STE, STC, S32T, S33T, S34T, S36T, a33, a34, a36 = constants
        S33C, S34C, S36C = crust_comp(S32C, constants)
        S32E = S32T - S32C
        S33E = S33T - S33C
        S34E = S34T - S34C
        S36E = S36T - S36C
        return S32C, S33C, S34C, S36C, S32E, S33E, S34E, S36E
    
    S32C = newton_raphson(guessx,constants,nr_tol,nr_step,f,df)
    result = reservoirs(S32C, constants)
   
    return result

####################
### CALCULATIONS ### 
####################

def iso_evo(t_step_Myr,end_time_Myr,nr_step,nr_tol,DM_mass_f,DM_mass_f_sil,DM_ST,DM_33d,DM_34d,DM_36d,M_mass_f,M_mass_f_sil,M_ST,M_33d,M_34d,M_36d,F_mass_f,F_ST,F_33d,F_34d,F_36d,SS_mass_f,SS_ST,SS_33d,SS_34d,SS_36d,S_ST,S_33d,S_34d,S_36d,rate_f,oscillate,resurf_cm_yr,sil_mag_S,thick_C,f_S2,f_pl2mo,f_sq,f_deep,f_remobilised,f_pu):
    
    print("start",datetime.datetime.now())
    time_step = t_step_Myr*60.*60.*24.*365.*1.e6 # time step in s
    end = (end_time_Myr+t_step_Myr)/t_step_Myr
    end = int(end)

    # rate mole S/s - plasma and photodissociation
    pd_r = kgs2mSs(calc_rate(rate_f["pd"],rate_kgs["pd_min"],rate_kgs["pd_max"]))
    pi_r = calc_rate(rate_f["pi"],rate_kgs["pi_min"],rate_kgs["pi_max"])
    ei_r = calc_rate(rate_f["ei"],rate_kgs["ei_min"],rate_kgs["ei_max"])
    ed_r = calc_rate(rate_f["ed"],rate_kgs["ed_min"],rate_kgs["ed_max"])
    ac_r = calc_rate(rate_f["ac"],rate_kgs["ac_min"],rate_kgs["ac_max"])
    rc_r = calc_rate(rate_f["rc"],rate_kgs["rc_min"],rate_kgs["rc_max"])
    ec_r = calc_rate(rate_f["ec"],rate_kgs["ec_min"],rate_kgs["ec_max"])
    
    # plasma reactions and fractionation
    pr_r = pi_r + ei_r + ed_r + ac_r + rc_r + ec_r
    rates_plasma = {"pr":pr_r,"pi":pi_r,"ei":ei_r,"ed":ed_r,"ac":ac_r,"rc":rc_r,"ec":ec_r}
    a33_plasma = {"pu":a33_pu,"gr":a33_gr,"th":a33_th,"ed":a33_ed,"ac":a33_ac,"ei":a33_ei,"rc":a33_rc,"ec":a33_ec,"pi":a33_pi}
    a34_plasma = {"pu":a34_pu,"gr":a34_gr,"th":a34_th,"ed":a34_ed,"ac":a34_ac,"ei":a34_ei,"rc":a34_rc,"ec":a34_ec,"pi":a34_pi}
    a36_plasma = {"pu":a36_pu,"gr":a36_gr,"th":a36_th,"ed":a36_ed,"ac":a36_ac,"ei":a36_ei,"rc":a36_rc,"ec":a36_ec,"pi":a36_pi}
    a33_pr = a_pr(a33_plasma,rates_plasma)
    a34_pr = a_pr(a34_plasma,rates_plasma)
    a36_pr = a_pr(a36_plasma,rates_plasma)
    pu_r = calc_rate(f_pu,rate_kgs["pu_min"],pr_r)
    pu_r = kgs2mSs(pu_r)
    pr_r = kgs2mSs(pr_r)
    pu_frac = pu_r/pr_r
    
    # mass of silicate crust and fraction of frost remobilised
    mass_sil_C = ((4./3.)*math.pi*((1822.6*1.e3)**3. - ((1822.6*1.e3)-thick_C)**3.))*sil_mag_rho # approx mass of crust

    ### INITIAL RESERVOIR NUMBERS ###
    # mantle
    if M_ST == "N":
        M_ST = initial_mass(M_mass_i_min,M_mass_i_max,M_mass_f)
    M_mass = initial_mass(M_mass_i_min_sil,M_mass_i_max_sil,M_mass_f_sil)
    M_mass = M_mass - mass_sil_C
    S_conc_M = S_conc(M_ST,M_mass)
    logM_ST, M_33R, M_34R, M_36R, M_32T_i, M_33T_i, M_34T_i, M_36T_i, M_33D, M_36D, M_32S, M_33S, M_34S, M_36S = initial_reservoir(M_ST,M_33d,M_34d,M_36d)
    
    # deep mantle
    if DM_ST == "N":
        if DM_mass_f != "N":
            DM_ST = initial_mass(M_mass_i_min,M_mass_i_max,DM_mass_f)
        else:
            DM_ST = 0.
    if DM_mass_f_sil != "N":
        DM_mass = initial_mass(M_mass_i_min_sil,M_mass_i_max_sil,DM_mass_f_sil)
    else:
        DM_mass = 0.
    logDM_ST, DM_33R, DM_34R, DM_36R, DM_32T_i, DM_33T_i, DM_34T_i, DM_36T_i, DM_33D, DM_36D, DM_32S, DM_33S, DM_34S, DM_36S = initial_reservoir(DM_ST,DM_33d,DM_34d,DM_36d)
    
    # crustal frosts
    if F_ST == "N":
        F_ST = initial_mass(F_mass_i_min,F_mass_i_max,F_mass_f) 
    logF_ST, F_33R, F_34R, F_36R, F_32T_i, F_33T_i, F_34T_i, F_36T_i, F_33D, F_36D, F_32S, F_33S, F_34S, F_36S = initial_reservoir(F_ST,F_33d,F_34d,F_36d)
    
    # crustal sulfates
    if SS_ST == "N":    
        SS_ST = initial_mass(SS_mass_i_min,SS_mass_i_max,SS_mass_f)
    logSS_ST, SS_33R, SS_34R, SS_36R, SS_32T_i, SS_33T_i, SS_34T_i, SS_36T_i, SS_33D, SS_36D, SS_32S, SS_33S, SS_34S, SS_36S = initial_reservoir(SS_ST,SS_33d,SS_34d,SS_36d)

    if S_ST == "N":
        S_ST = 0.
    logS_ST, S_33R, S_34R, S_36R, S_32T_i, S_33T_i, S_34T_i, S_36T_i, S_33D, S_36D, S_32S, S_33S, S_34S, S_36S = initial_reservoir(S_ST,S_33d,S_34d,S_36d)
    
    # mass balance
    Io_ST_initial = reservoir_totals(M_ST,DM_ST,F_ST,SS_ST,S_ST)        
    Io_32S_initial = reservoir_totals(M_32S,DM_32S,F_32S,SS_32S,S_32S)
    Io_33S_initial = reservoir_totals(M_33S,DM_33S,F_33S,SS_33S,S_33S)
    Io_34S_initial = reservoir_totals(M_34S,DM_34S,F_34S,SS_34S,S_34S)
    Io_36S_initial = reservoir_totals(M_36S,DM_36S,F_36S,SS_36S,S_36S)
    mbST = mass_balance(Io_ST_initial,Io_ST_initial)
    mb32S = mass_balance(Io_32S_initial,Io_32S_initial)
    mb33S = mass_balance(Io_33S_initial,Io_33S_initial)
    mb34S = mass_balance(Io_34S_initial,Io_34S_initial)
    mb36S = mass_balance(Io_36S_initial,Io_36S_initial)
    
    ## initial step ##
    n = 0.
    t_s = n*time_step
    t_s_Myr = n*t_step_Myr

    # set up results table
    results = pd.DataFrame([["rate factor pd","rate factor pi","rate factor ei","rate factor ed","rate factor ac",
                         "rate factor rc", "rate factor ec","resurfacing rate (cm/yr)","sil mag S (wf)", "crustal thickness (m)",
                         "fraction of f_crust_returned that is remolised","S2 to SO2","plutons from mantle melting","sulfate sequestration","deep mantle - fraction of mantle melting","mass silicate mantle"]])
    results2 = pd.DataFrame([[rate_f["pd"],rate_f["pi"],rate_f["ei"],rate_f["ed"],rate_f["ac"],rate_f["rc"], 
                        rate_f["ec"],resurf_cm_yr,sil_mag_S, thick_C,f_remobilised,f_S2,f_pl2mo,f_sq,f_deep,M_mass]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([["time step","time (Myr)","[S] mantle",
                         "d34S M","d34S F","d34S SS","d34S S","d34S DM","d34S og","d34S iS",
                         "d33S M","d33S F","d33S SS", "d33S S","d33S DM","d33S og","d33S iS",
                         "d36S M","d36S F","d36S SS","d36S S","d36S DM","d36S og","d36S iS",
                         "D33S M","D33S F","D33S SS","D33S S","D33S DM","D33S og","D33S iS",
                         "D36S M","D36S F","D36S SS","D36S S","D36S DM","D36S og","D36S iS",
                         "ST M","ST F", "ST SS","ST S","ST DM",
                         "log ST M","log ST F", "log ST SS","log ST S","log ST DM",
                         "F mo", "F fr", "F pr","F pu", "F sn", "F pd", "F hg", "F bu", "F pl","F rs","F sq","F ao","F dm",
                         "mo/mo+ca", "bu/bu+pl", 
                         "mbST","mb32","mb33","mb34","mb36","",
                         "R33 M", "R33 F", "R33 SS","R33 S","R33 DM","R33 og","R33 iS",
                         "R34 M", "R34 F", "R34 SS","R34 S","R34 DM","R34 og","R34 iS",
                         "R36 M", "R36 F", "R36 SS","R36 S","R36 DM","R36 og","R36 iS",
                         "a33_bu", "a34_bu", "a36_bu",
                         "a33_escape-burial", "a34_escape-burial", "a36_escape-burial"]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([[n,t_s,S_conc_M,
                         M_34d,F_34d,SS_34d,S_34d,DM_34d,"","",
                         M_33d,F_33d,SS_33d,S_33d,DM_33d,"","",
                         M_36d,F_36d,SS_36d,S_36d,DM_36d,"","",
                         M_33D,F_33D,SS_33D,S_33D,DM_33D,"","",
                         M_36D,F_36D,SS_36D,S_36D,DM_36D,"","",
                         M_ST,F_ST,SS_ST,S_ST,DM_ST,
                         logM_ST,logF_ST,logSS_ST,logS_ST,logDM_ST,
                         "","","","","","","","","","","","","",
                         "","",
                         mbST,mb32S,mb33S,mb34S,mb36S,
                         "",
                         M_33R, F_33R, SS_33R, S_33R, DM_33R,"","",
                         M_34R, F_34R, SS_34R, S_34R, DM_36R,"","",
                         M_36R, F_36R, SS_36R, S_36R, DM_36R,"","",
                         "","","","","","","","",""]])         
    results = pd.concat([results, results2], ignore_index=True)
    results.to_csv('time_evolution.csv', index=False, header=False)

    for n in range(1,end,1): 
        t_s = n*time_step
        t_s_Myr = n*t_step_Myr
        if(n % 1000==0):
            print(t_s_Myr,F_34d,datetime.datetime.now())
    
        if oscillate == "Y":
            if t_s_Myr >= 4400.:
                resurf_cm_yr = 5.
            elif t_s_Myr >= 100. and t_s_Myr < 200. or t_s_Myr >= 300. and t_s_Myr < 400. or t_s_Myr >= 500. and t_s_Myr < 600. or t_s_Myr >= 700. and t_s_Myr < 800. or t_s_Myr >= 900. and t_s_Myr < 1000. or t_s_Myr >= 1100. and t_s_Myr < 1200.  or t_s_Myr >= 1300. and t_s_Myr < 1400. or t_s_Myr >= 1500. and t_s_Myr < 1600. or t_s_Myr >= 1700. and t_s_Myr < 1800.  or t_s_Myr >= 1900. and t_s_Myr < 2000. or t_s_Myr >= 2100. and t_s_Myr < 2200. or t_s_Myr >= 2300. and t_s_Myr < 2400. or t_s_Myr >= 2500. and t_s_Myr < 2600. or t_s_Myr >= 2700. and t_s_Myr < 2800. or t_s_Myr >= 2900. and t_s_Myr < 3000. or t_s_Myr >= 3100. and t_s_Myr < 3200. or t_s_Myr >= 3300. and t_s_Myr < 3400. or t_s_Myr >= 3500. and t_s_Myr < 3600. or t_s_Myr >= 3700. and t_s_Myr < 3800. or t_s_Myr >= 3900. and t_s_Myr < 4000. or t_s_Myr >= 4100. and t_s_Myr < 4200. or t_s_Myr >= 4300. and t_s_Myr < 4400. :
                resurf_cm_yr = 1.
            elif t_s_Myr < 100. or t_s_Myr >= 200. and t_s_Myr < 300. or t_s_Myr >= 400. and t_s_Myr < 500. or t_s_Myr >= 600. and t_s_Myr < 700. or t_s_Myr >= 800. and t_s_Myr < 900. or t_s_Myr >= 1000. and t_s_Myr < 1100. or t_s_Myr >= 1200. and t_s_Myr < 1300. or t_s_Myr >= 1400. and t_s_Myr < 1500. or t_s_Myr >= 1600. and t_s_Myr < 1700. or t_s_Myr >= 1800. and t_s_Myr < 1900. or t_s_Myr >= 2000. and t_s_Myr < 2100. or t_s_Myr >= 2200. and t_s_Myr < 2300. or t_s_Myr >= 2400. and t_s_Myr < 2500. or t_s_Myr >= 2600. and t_s_Myr < 2700. or t_s_Myr >= 2800. and t_s_Myr < 2900. or t_s_Myr >= 3000. and t_s_Myr < 3100. or t_s_Myr >= 3200. and t_s_Myr < 3300. or t_s_Myr >= 3400. and t_s_Myr < 3500. or t_s_Myr >= 3600. and t_s_Myr < 3700. or t_s_Myr >= 3800. and t_s_Myr < 3900. or t_s_Myr >= 4000. and t_s_Myr < 4100. or t_s_Myr >= 4200. and t_s_Myr < 4300.:
                resurf_cm_yr = 9.
    
        # mantle melting
        sil_mag_m_yr = (resurf_cm_yr/(1.-f_pl2mo))*0.01 # silicate magmatism in m/yr
        sil_mag_m_s = sil_mag_m_yr/(365.*24.*60.*60.) # silicate magmatism in m/s
        sil_mag_kg_s = ((sil_mag_m_s*Io_SA)*sil_mag_rho) # silicate magmatism mass in kg/s
        f_crust_return = (sil_mag_m_s*time_step)/thick_C # volume fraction crust returned in a time step
        f_remob = f_remobilised*f_crust_return # fraction of crustal frosts that are remobilised

        S_conc_M = S_conc(M_ST,M_mass)
        if S_conc_M > sil_mag_S:
            mm_r = molSs_mm(sil_mag_kg_s,sil_mag_S,1.)
        else:
            mm_r = molSs_mm(sil_mag_kg_s,S_conc_M,1.)
    
        if DM_ST > 0.:
            S_conc_DM = S_conc(DM_ST,DM_mass)
            if S_conc_DM > sil_mag_S:
                dm_r = molSs_mm(sil_mag_kg_s,sil_mag_S,f_deep)
            else:
                dm_r =  molSs_mm(sil_mag_kg_s,S_conc_DM,f_deep)
        else:
            dm_r = 0.
    
        # rates
        mo_r = (1.-f_pl2mo)*mm_r
        fr_r = (f_remob*F_ST)/time_step
        hg_r = (mo_r+fr_r)*f_S2
        sq_r = (mo_r+fr_r)*f_sq
        ao_r = (mo_r+fr_r) - (sq_r+hg_r) # atmospheric outgassing rate
        sn_r = ao_r-(pu_r+pd_r)
        
        pl_r = mm_r*f_pl2mo
        rs_r = (f_crust_return*SS_ST)/time_step

        # flux for snow...
        if sn_r < 0.:
            bu_r = pd_r
        else:    
            bu_r = pd_r + sn_r

        # flux
        dm_F = dm_r*time_step
        mm_F = mm_r*time_step
        mo_F = mo_r*time_step
        fr_F = fr_r*time_step
        pd_F = pd_r*time_step
        hg_F = hg_r*time_step
        sq_F = sq_r*time_step
        ao_F = ao_r*time_step
        sn_F = sn_r*time_step
        bu_F = bu_r*time_step
        pl_F = pl_r*time_step
        pr_F = pr_r*time_step
        pu_F = pu_r*time_step
        rs_F = rs_r*time_step
    
        # fractions
        mo_mo_ca = mo_r/(mo_r+fr_r) # amount of volcanic S from mantle outgassing vs. total (inc. frost remobilisation)
        bu_bu_pl = bu_r/(bu_r+pl_r+hg_r+sq_r) # amout of S lost from atmosphere to surface vs total (inc. space loss)
    
        # burial
        if sn_r > 0.:
            a33_bu = a_bu(sn_F, pd_F, a33_vp, a33_pd)
            a34_bu = a_bu(sn_F, pd_F, a34_vp, a34_pd)
            a36_bu = a_bu(sn_F, pd_F, a36_vp, a36_pd)
        else:
            a33_bu = a_bu(0., pd_F, a33_vp, a33_pd)
            a34_bu = a_bu(0., pd_F, a34_vp, a34_pd)
            a36_bu = a_bu(0., pd_F, a36_vp, a36_pd)

        # escape-burial
        a33_eb = a33_pr/a33_bu
        a34_eb = a34_pr/a34_bu
        a36_eb = a36_pr/a36_bu

        # mo = mantle outgassing
        mo_33R, mo_34R, mo_36R, mo_32S, mo_33S, mo_34S, mo_36S = process_R_S(mo_F,M_33R,M_34R,M_36R,a33_mo,a34_mo,a36_mo)
        # rf = crustal remobilisation     
        fr_33R, fr_34R, fr_36R, fr_32S, fr_33S, fr_34S, fr_36S = process_R_S(fr_F,F_33R,F_34R,F_36R,a33_fr,a34_fr,a36_fr)
        # dm = deep mantle mixing
        dm_33R, dm_34R, dm_36R, dm_32S, dm_33S, dm_34S, dm_36S = process_R_S(dm_F,DM_33R,DM_34R,DM_36R,a33_dm,a34_dm,a36_dm)
        
        # ti = total input
        ti_32S = mo_32S + fr_32S
        ti_33S = mo_33S + fr_33S
        ti_34S = mo_34S + fr_34S
        ti_36S = mo_36S + fr_36S
        ti_TS = ti_32S + ti_33S + ti_34S + ti_36S
        ti_34R = ti_34S/ti_32S
        ti_33R = ti_33S/ti_32S
        ti_36R = ti_36S/ti_32S

        # hg = homogeneous gas
        hg_33R, hg_34R, hg_36R, hg_32S, hg_33S, hg_34S, hg_36S = process_R_S(hg_F,ti_33R,ti_34R,ti_36R,a33_hg,a34_hg,a36_hg)
        # sq = sulfate sequestration
        sq_33R, sq_34R, sq_36R, sq_32S, sq_33S, sq_34S, sq_36S = process_R_S(sq_F,ti_33R,ti_34R,ti_36R,a33_sq,a34_sq,a36_sq)
        # sn = if snow is negative, adding frost to atmosphere...
        if sn_r < 0.:
            sn_33R, sn_34R, sn_36R, sn_32S, sn_33S, sn_34S, sn_36S = process_R_S((-1.*sn_F),F_33R,F_34R,F_36R,a33_vp,a34_vp,a36_vp)
        else:
            sn_32S, sn_33S, sn_34S, sn_36S = 0., 0., 0., 0.

        # ao = atmospheric outgassing
        ao_32S = ti_32S - (hg_32S + sq_32S) + sn_32S
        ao_33S = ti_33S - (hg_33S + sq_33S) + sn_33S
        ao_34S = ti_34S - (hg_34S + sq_34S) + sn_34S
        ao_36S = ti_36S - (hg_36S + sq_36S) + sn_36S
        ao_33R, ao_36R, ao_34R, ao_33d, ao_34d, ao_36d, ao_33D, ao_36D, ao_TS, logao_ST = reservoir_isotope(ao_32S, ao_33S, ao_34S, ao_36S)
    
        # assuming space loss material and burial material have correct alpha between them
        constants = ao_TS, pu_F, bu_F, ao_32S, ao_33S, ao_34S, ao_36S, a33_eb, a34_eb, a36_eb
        if n == 1:
            reservoirs = fractionate(constants,nr_step,nr_tol,ao_32S)
        else:
            reservoirs = fractionate(constants,nr_step,nr_tol,S32C)
        S32C, S33C, S34C, S36C, S32E, S33E, S34E, S36E = reservoirs
        
        # fractionating space loss and photo-dissociation, and remainder going to frost... ONLY WORKS IF PD = 0
        # pu_33R, pu_34R, pu_36R, pu_32S, pu_33S, pu_34S, pu_36S = process_R_S(pu_F,ao_33R,ao_34R,ao_36R,a33_pr,a34_pr,a36_pr)
        # sn_32S, sn_33S, sn_34S, sn_36S = (ao_32S - pu_32S), (ao_33S - pu_33S), (ao_34S - pu_34S), (ao_36S - pu_36S) 
        # S32E, S33E, S34E, S36E = pu_32S, pu_33S, pu_34S, pu_36S
        # S32C, S33C, S34C, S36C = sn_32S, sn_33S, sn_34S, sn_36S
        # E_33R, E_34R, E_36R, E_33d, E_34d, E_36d, E_33D, E_36D, E_ST, logE_ST = reservoir_isotope(S32E, S33E, S34E, S36E)
        # C_33R, C_34R, C_36R, C_33d, C_34d, C_36d, C_33D, C_36D, C_ST, logC_ST  = reservoir_isotope(S32C, S33C, S34C, S36C)
        #undo_32S, undo_33S, undo_34S, undo_36S = (S32E + S32C), (S33E + S33C), (S34E + S34C), (S36E + S36C)
        #undo_ST,undo_34R, undo_33R, undo_36R, logundo_ST, undo_33d, undo_34d, undo_36d, undo_33D, undo_36D = reservoir_isotope(undo_32S, undo_33S, undo_34S, undo_36S)
        #print(ao_33D, ao_36D,E_33D, E_36D, C_33D, C_36D,undo_33D, undo_36D)

        # instantaneous space
        iS_33R, iS_34R, iS_36R, iS_33d, iS_34d, iS_36d, iS_33D, iS_36D, iS_TS, logiS_ST = reservoir_isotope(S32E, S33E, S34E, S36E)
    
        # pl = pluton
        pl_33R, pl_34R, pl_36R, pl_32S, pl_33S, pl_34S, pl_36S = process_R_S(pl_F,M_33R,M_34R,M_36R,a33_pl,a34_pl,a36_pl)
        # rs = return silicate/sulfate
        rs_33R, rs_34R, rs_36R, rs_32S, rs_33S, rs_34S, rs_36S = process_R_S(rs_F,SS_33R,SS_34R,SS_36R,a33_rm,a34_rm,a36_rm)

        # mantle
        M_32S = mantle(M_32S, mo_32S, pl_32S, rs_32S, dm_32S)
        M_33S = mantle(M_33S, mo_33S, pl_33S, rs_33S, dm_33S)
        M_34S = mantle(M_34S, mo_34S, pl_34S, rs_34S, dm_34S)
        M_36S = mantle(M_36S, mo_36S, pl_36S, rs_36S, dm_36S)
        M_33R, M_34R, M_36R, M_33d, M_34d, M_36d, M_33D, M_36D, M_ST, logM_ST = reservoir_isotope(M_32S, M_33S, M_34S, M_36S)

        # deep mantle
        DM_32S = deep_mantle(DM_32S, dm_32S)
        DM_33S = deep_mantle(DM_33S, dm_33S)
        DM_34S = deep_mantle(DM_34S, dm_34S)
        DM_36S = deep_mantle(DM_36S, dm_36S)
        DM_33R, DM_34R, DM_36R, DM_33d, DM_34d, DM_36d, DM_33D, DM_36D, DM_ST, logDM_ST = reservoir_isotope(DM_32S, DM_33S, DM_34S, DM_36S)

        # frost
        if sn_F < 0.:
            fr_32S = fr_32S + sn_32S
            fr_33S = fr_33S + sn_33S
            fr_34S = fr_34S + sn_34S
            fr_36S = fr_36S + sn_36S
            
        F_32S = frost(F_32S,fr_32S,S32C,hg_32S)
        F_33S = frost(F_33S,fr_33S,S33C,hg_33S)
        F_34S = frost(F_34S,fr_34S,S34C,hg_34S)
        F_36S = frost(F_36S,fr_36S,S36C,hg_36S)
        F_33R, F_34R, F_36R, F_33d, F_34d, F_36d, F_33D, F_36D, F_ST, logF_ST = reservoir_isotope(F_32S, F_33S, F_34S, F_36S)

        # SS = silicate + sulfate
        SS_32S = silsulf(SS_32S,rs_32S,sq_32S,pl_32S)
        SS_33S = silsulf(SS_33S,rs_33S,sq_33S,pl_33S)
        SS_34S = silsulf(SS_34S,rs_34S,sq_34S,pl_34S)
        SS_36S = silsulf(SS_36S,rs_36S,sq_36S,pl_36S)
        SS_33R, SS_34R, SS_36R, SS_33d, SS_34d, SS_36d, SS_33D, SS_36D, SS_ST, logSS_ST, = reservoir_isotope(SS_32S, SS_33S, SS_34S, SS_36S)

        # S = space 
        S_32S = space(S_32S,S32E)
        S_33S = space(S_33S,S33E)
        S_34S = space(S_34S,S34E)
        S_36S = space(S_36S,S36E)
        S_33R, S_34R, S_36R, S_33d, S_34d, S_36d, S_33D, S_36D, S_ST, logS_ST = reservoir_isotope(S_32S, S_33S, S_34S, S_36S)
    
        # check mass balance
        Io_ST = reservoir_totals(M_ST,DM_ST,F_ST,SS_ST,S_ST)        
        Io_32S = reservoir_totals(M_32S,DM_32S,F_32S,SS_32S,S_32S)
        Io_33S = reservoir_totals(M_33S,DM_33S,F_33S,SS_33S,S_33S)
        Io_34S = reservoir_totals(M_34S,DM_34S,F_34S,SS_34S,S_34S)
        Io_36S = reservoir_totals(M_36S,DM_36S,F_36S,SS_36S,S_36S)
        mbST = mass_balance(Io_ST_initial,Io_ST)
        mb32S = mass_balance(Io_32S_initial,Io_32S)
        mb33S = mass_balance(Io_33S_initial,Io_33S)
        mb34S = mass_balance(Io_34S_initial,Io_34S)
        mb36S = mass_balance(Io_36S_initial,Io_36S)
    
        results2 = pd.DataFrame([[n,t_s_Myr,S_conc_M,
                         M_34d,F_34d,SS_34d,S_34d,DM_34d,ao_34d,iS_34d,
                         M_33d,F_33d,SS_33d,S_33d,DM_33d,ao_33d,iS_33d,
                         M_36d,F_36d,SS_36d,S_36d,DM_36d,ao_36d,iS_36d,
                         M_33D,F_33D,SS_33D,S_33D,DM_33D,ao_33D,iS_33D,
                         M_36D,F_36D,SS_36D,S_36D,DM_36D,ao_36D,iS_36D,
                         M_ST,F_ST,SS_ST,S_ST,DM_ST,
                         logM_ST,logF_ST,logSS_ST,logS_ST,logDM_ST,
                         mo_F,fr_F,pr_F,pu_F,sn_F,pd_F,hg_F,bu_F,pl_F,rs_F,sq_F,ao_F,dm_F,
                         mo_mo_ca, bu_bu_pl, 
                         mbST,mb32S,mb33S,mb34S,mb36S,
                         "",
                         M_33R, F_33R, SS_33R, S_33R,DM_33R,ao_33R,iS_33R,
                         M_34R, F_34R, SS_34R, S_34R,DM_34R,ao_34R,iS_34R,
                         M_36R, F_36R, SS_36R, S_36R,DM_36R,ao_36R,iS_36R,
                         a33_bu,a34_bu,a36_bu,a33_eb,a34_eb,a36_eb]]) 
        results = pd.concat([results, results2], ignore_index=True)
        if(n % 1000==0):
            results.to_csv('time_evolution.csv', index=False, header=False)
        if(n==(end-1.)):
            results.to_csv('time_evolution.csv', index=False, header=False)
            print("end",t_s_Myr,F_34d,datetime.datetime.now())

###############################            
### RAYLEIGHT BUT BOX MODEL ###  
###############################

def iso_evo_simple(t_step_Myr,end_time_Myr,nr_step,nr_tol,DM_mass_f,DM_mass_f_sil,DM_ST,DM_33d,DM_34d,DM_36d,M_mass_f,M_mass_f_sil,M_ST,M_33d,M_34d,M_36d,F_mass_f,F_ST,F_33d,F_34d,F_36d,SS_mass_f,SS_ST,SS_33d,SS_34d,SS_36d,S_ST,S_33d,S_34d,S_36d,rate_f,oscillate,resurf_cm_yr,sil_mag_S,thick_C,f_S2,f_pl2mo,f_sq,f_deep,f_remobilised,f_pu):
    
    print("start",datetime.datetime.now())
    time_step = t_step_Myr*60.*60.*24.*365.*1.e6 # time step in s
    end = (end_time_Myr+t_step_Myr)/t_step_Myr
    end = int(end)

    # rate mole S/s - plasma and photodissociation
    pi_r = calc_rate(rate_f["pi"],rate_kgs["pi_min"],rate_kgs["pi_max"])
    ei_r = calc_rate(rate_f["ei"],rate_kgs["ei_min"],rate_kgs["ei_max"])
    ed_r = calc_rate(rate_f["ed"],rate_kgs["ed_min"],rate_kgs["ed_max"])
    ac_r = calc_rate(rate_f["ac"],rate_kgs["ac_min"],rate_kgs["ac_max"])
    rc_r = calc_rate(rate_f["rc"],rate_kgs["rc_min"],rate_kgs["rc_max"])
    ec_r = calc_rate(rate_f["ec"],rate_kgs["ec_min"],rate_kgs["ec_max"])
    
    # plasma reactions and fractionation
    pr_r = pi_r + ei_r + ed_r + ac_r + rc_r + ec_r
    pu_r = calc_rate(f_pu,rate_kgs["pu_min"],pr_r)
    pu_r = kgs2mSs(pu_r)
    pr_r = kgs2mSs(pr_r)
    
    # mass of silicate crust and fraction of frost remobilised
    mass_sil_C = ((4./3.)*math.pi*((1822.6*1.e3)**3. - ((1822.6*1.e3)-thick_C)**3.))*sil_mag_rho # approx mass of crust

    ### INITIAL RESERVOIR NUMBERS ###
    # mantle
    if M_ST == "N":
        M_ST = M_mass_i_min + M_mass_f*(M_mass_i_max - M_mass_i_min)
    M_mass = (M_mass_i_min_sil + M_mass_f_sil*(M_mass_i_max_sil - M_mass_i_min_sil)) - mass_sil_C
    S_conc_M = ((M_ST*32.)/1000.)/M_mass
    logM_ST, M_33R, M_34R, M_36R, M_32T_i, M_33T_i, M_34T_i, M_36T_i, M_33D, M_36D, M_32S, M_33S, M_34S, M_36S = initial_reservoir(M_ST,M_33d,M_34d,M_36d)
        
    if S_ST == "N":
        S_ST = 0.
    logS_ST, S_33R, S_34R, S_36R, S_32T_i, S_33T_i, S_34T_i, S_36T_i, S_33D, S_36D, S_32S, S_33S, S_34S, S_36S = initial_reservoir(S_ST,S_33d,S_34d,S_36d)

    Io_ST_initial = M_ST + S_ST
    mb = Io_ST_initial - Io_ST_initial

    ## initial step ##
    n = 0.
    t_s = n*time_step
    t_s_Myr = n*t_step_Myr

    # set up results table
    results = pd.DataFrame([["rate factor pd","rate factor pi","rate factor ei","rate factor ed","rate factor ac",
                         "rate factor rc", "rate factor ec","resurfacing rate (cm/yr)","sil mag S (wf)", "crustal thickness (m)",
                         "fraction of f_crust_returned that is remolised","S2 to SO2","plutons from mantle melting","sulfate sequestration","deep mantle - fraction of mantle melting","mass silicate mantle"]])
    results2 = pd.DataFrame([[rate_f["pd"],rate_f["pi"],rate_f["ei"],rate_f["ed"],rate_f["ac"],rate_f["rc"], 
                        rate_f["ec"],resurf_cm_yr,sil_mag_S, thick_C,f_remobilised,f_S2,f_pl2mo,f_sq,f_deep,M_mass]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([["time step","time (Myr)","[S] mantle",
                         "d34S M","d34S F","d34S SS","d34S S","d34S DM","d34S og","d34S iS",
                         "d33S M","d33S F","d33S SS", "d33S S","d33S DM","d33S og","d33S iS",
                         "d36S M","d36S F","d36S SS","d36S S","d36S DM","d36S og","d36S iS",
                         "D33S M","D33S F","D33S SS","D33S S","D33S DM","D33S og","D33S iS",
                         "D36S M","D36S F","D36S SS","D36S S","D36S DM","D36S og","D36S iS",
                         "ST M","ST F", "ST SS","ST S","ST DM",
                         "log ST M","log ST F", "log ST SS","log ST S","log ST DM",
                         "F mo", "F fr", "F pr","F pu", "F sn", "F pd", "F hg", "F bu", "F pl","F rs","F sq","F ao","F dm",
                         "mo/mo+ca", "bu/bu+pl", "mass balance check","",
                         "R33 M", "R33 F", "R33 SS","R33 S","R33 DM","R33 og","R33 iS",
                         "R34 M", "R34 F", "R34 SS","R34 S","R34 DM","R34 og","R34 iS",
                         "R36 M", "R36 F", "R36 SS","R36 S","R36 DM","R36 og","R36 iS",
                         "a33_bu", "a34_bu", "a36_bu",
                         "a33_escape-burial", "a34_escape-burial", "a36_escape-burial"]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([[n,t_s,S_conc_M,
                         M_34d,"","",S_34d,"","","",
                         M_33d,"","",S_33d,"","","",
                         M_36d,"","",S_36d,"","","",
                         M_33D,"","",S_33D,"","","",
                         M_36D,"","",S_36D,"","","",
                         M_ST,0.,0.,S_ST,0.,
                         logM_ST,"","",logS_ST,"",
                         "","","","","","","","","","","","","",
                         "","",mb,"",
                         M_33R, "", "", S_33R, "","","",
                         M_34R, "", "", S_34R, "","","",
                         M_36R, "", "", S_36R, "","","",
                         "","","","","","","","",""]])   
    results = pd.concat([results, results2], ignore_index=True)      
    results.to_csv('time_evolution.csv', index=False, header=False)

    for n in range(1,end,1): 
        t_s = n*time_step
        t_s_Myr = n*t_step_Myr
        if(n % 1000==0):
            print(t_s_Myr,F_34d,datetime.datetime.now())
        
        # flux
        pr_F = pr_r*time_step
        pu_F = pu_r*time_step
        
        # iS = instananeous space loss
        iS_33R, iS_34R, iS_36R, iS_32S, iS_33S, iS_34S, iS_36S = process_R_S(pu_F,M_33R,M_34R,M_36R,a33_gr,a34_gr,a36_gr)
        iS_33R, iS_34R, iS_36R, iS_33d, iS_34d, iS_36d, iS_33D, iS_36D, iS_ST, logiS_ST = reservoir_isotope(iS_32S, iS_33S, iS_34S, iS_36S)
        
        # mantle
        M_32S = mantle(M_32S, iS_32S, 0., 0., 0.)
        M_33S = mantle(M_33S, iS_33S, 0., 0., 0.)
        M_34S = mantle(M_34S, iS_34S, 0., 0., 0.)
        M_36S = mantle(M_36S, iS_36S, 0., 0., 0.)
        M_33R, M_34R, M_36R, M_33d, M_34d, M_36d, M_33D, M_36D, M_ST, logM_ST = reservoir_isotope(M_32S, M_33S, M_34S, M_36S)

        # S = space 
        S_32S = space(S_32S,iS_32S)
        S_33S = space(S_33S,iS_33S)
        S_34S = space(S_34S,iS_34S)
        S_36S = space(S_36S,iS_36S)
        S_33R, S_34R, S_36R, S_33d, S_34d, S_36d, S_33D, S_36D, S_ST, logS_ST = reservoir_isotope(S_32S, S_33S, S_34S, S_36S)
    
        # check mass balance
        mb = (Io_ST_initial - M_ST - S_ST)/Io_ST_initial
    
        results2 = pd.DataFrame([[n,t_s_Myr,S_conc_M,
                         M_34d,"","",S_34d,"","",iS_34d,
                         M_33d,"","",S_33d,"","",iS_33d,
                         M_36d,"","",S_36d,"","",iS_36d,
                         M_33D,"","",S_33D,"","",iS_33D,
                         M_36D,"","",S_36D,"","",iS_36D,
                         M_ST,0.,0.,S_ST,0.,
                         logM_ST,"","",logS_ST,"",
                         0.,0.,pr_F,pu_F,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                         "", "", mb, "",
                         M_33R, "", "", S_33R,"","",iS_33R,
                         M_34R, "", "", S_34R,"","",iS_34R,
                         M_36R, "", "", S_36R,"","",iS_36R,
                         "","","",a33_gr,a34_gr,a36_gr]]) 
        results = pd.concat([results, results2], ignore_index=True)
        if(n % 1000==0):
            results.to_csv('time_evolution.csv', index=False, header=False)
        if(n==(end-1.)):
            results.to_csv('time_evolution.csv', index=False, header=False)
            print("end",t_s_Myr,F_34d,datetime.datetime.now())
            

##########################       
### atmosphere profile ###
##########################
def calc_atm(profile,F_33R,F_34R,F_36R):

    A0_33R, A0_34R, A0_36R = F_33R*a33_vp, F_34R*a34_vp, F_36R*a36_vp
    A0_33d, A0_34d, A0_36d = R2d(A0_33R,33,VCDT), R2d(A0_34R,34,VCDT), R2d(A0_36R,36,VCDT)
    A0_33D, A0_36D = MIF(A0_33R,A0_34R,A0_36R)

    end = 615

    # constants
    k = 1.38064852e-23
    S32 = 31.972071 # mass of 32S in Da
    S33 = 32.9714589099 # mass of 33S in Da
    S34 = 33.967867 # mass of 34S in Da
    S36 = 35.9670807 # mass of 36S in Da
    Da = 1.66054E-27 # mass of 1 Da in kg
    DS_33 = (S32 - S33)*Da
    DS_34 = (S32 - S34)*Da
    DS_36 = (S32 - S36)*Da
    hp = 34. # homopause in km

    # set up results table
    results = pd.DataFrame([["F_33R", "F_34R", "F_36R","33a_vp","34a_vp","36a_vp"]])
    results2 = pd.DataFrame([[F_33R, F_34R, F_36R,a33_vp,a34_vp,a36_vp]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([["Altitude (km)", "T (K)", "33a","34a","36a","33R","34R","36R",
                         "d33S","d34S","d36S","D33S","D36S"]])
    results = pd.concat([results, results2], ignore_index=True)
    results2 = pd.DataFrame([[profile.loc[0,"H"],profile.loc[0,"T"],"","","",A0_33R,A0_34R,A0_36R,
                          A0_33d,A0_34d,A0_36d,A0_33D, A0_36D]])
    results = pd.concat([results, results2], ignore_index=True)
    results.to_csv('atmosphere.csv', index=False, header=False)

    for n in range(1,end,1): 
    
        H = profile.loc[n,"H"] 
        T = profile.loc[n,"T"]
        g = ((0.000000000066743)*(8.95*(10.**22)))/(((1822.7+H)*1000.)**2.)
    
        if H > hp:
            a33 = math.exp((g*DS_33*1000.*H)/(k*T))
            a34 = math.exp((g*DS_34*1000.*H)/(k*T))
            a36 = math.exp((g*DS_36*1000.*H)/(k*T))
        else:
            a33 = 1.
            a34 = 1.
            a36 = 1.
        A2_33R, A2_34R, A2_36R = A0_33R*a33, A0_34R*a34, A0_36R*a36
        A2_33d, A2_34d, A2_36d = R2d(A2_33R,33,VCDT), R2d(A2_34R,34,VCDT), R2d(A2_36R,36,VCDT)
        A2_33D, A2_36D = MIF(A2_33R,A2_34R,A2_36R)
        results2 = pd.DataFrame([[profile.loc[n,"H"],profile.loc[n,"T"],a33,a34,a36,A2_33R,A2_34R,A2_36R,
                          A2_33d,A2_34d,A2_36d,A2_33D, A2_36D]])
        results = pd.concat([results, results2], ignore_index=True)
        results.to_csv('atmosphere.csv', index=False, header=False)
    
### range of fractionation factors at a single step ###
def range_frac_fac_ss():

    # create results table
    results = pd.DataFrame([["a33_vp","a33_pd","a33_hg","a33_sl","a34_vp","a34_pd","a34_hg","a34_sl",
                         "a36_vp","a36_pd","a36_hg","a36_sl"]])
    results1 = pd.DataFrame([[a33_vp,a33_pd,a33_hg,a33_sl,a34_vp,a34_pd,a34_hg,a34_sl,
                         a36_vp,a36_pd,a36_hg,a36_sl]])
    results = pd.concat([results, results1], ignore_index=True)
    results1 = pd.DataFrame([["x_sn","x_pd","x_hg","a33_bu","a34_bu","a36_bu","a33_sb","a34_sb","a36_sb",
                         "D33_bu","D34_bu","D36_bu","D33_sb","D34_sb","D36_sb"]])
    results = pd.concat([results, results1], ignore_index=True)

    for n in range(0,100,1):
        x_sn = n/100.
        for m in range (0,(100-n),1):
            x_pd = m/100.
            x_hg = 1.-x_pd-x_sn
            a33_bu = a_bu(x_sn, x_pd, x_hg, a33_vp, a33_pd, a33_hg)
            a34_bu = a_bu(x_sn, x_pd, x_hg, a34_vp, a34_pd, a34_hg)
            a36_bu = a_bu(x_sn, x_pd, x_hg, a36_vp, a36_pd, a36_hg)
            D33_bu, D34_bu, D36_bu = (1.-a33_bu)*1000.,(1.-a34_bu)*1000.,(1.-a36_bu)*1000.
            a33_sb = a33_sl/a33_bu
            a34_sb = a34_sl/a34_bu
            a36_sb = a36_sl/a36_bu
            D33_sb, D34_sb, D36_sb = (1.-a33_sb)*1000.,(1.-a34_sb)*1000.,(1.-a36_sb)*1000.
            results1 = pd.DataFrame([[x_sn,x_pd,x_hg,a33_bu,a34_bu,a36_bu,a33_sb,a34_sb,a36_sb,
                                 D33_bu,D34_bu,D36_bu,D33_sb,D34_sb,D36_sb]])
            results = pd.concat([results, results1], ignore_index=True)
            results.to_csv('a_at_single_step.csv', index=False, header=False)  
            
### range of fractionation factors for various proportions of space to burial ###
def range_frac_fac_s2b(x_hg):

    # create results table
    results = pd.DataFrame([["a33_vp","a33_pd","a33_hg","a33_sl","a34_vp","a34_pd","a34_hg","a34_sl",
                         "a36_vp","a36_pd","a36_hg","a36_sl","x_hg"]])
    results1 = pd.DataFrame([[a33_vp,a33_pd,a33_hg,a33_sl,a34_vp,a34_pd,a34_hg,a34_sl,
                         a36_vp,a36_pd,a36_hg,a36_sl,x_hg]])
    results = pd.concat([results, results1], ignore_index=True)
    results1 = pd.DataFrame([["x_sl","x_bu","x_sn","x_pd","x_hg","a33_sb","a34_sb","a36_sb","D33_sb","D34_sb","D36_sb"]])
    results = pd.concat([results, results1], ignore_index=True)

    for n in range(0,80,1):
        x_sn = n/100.
        x_pd = 1.-x_hg-x_sn
        a33_bu = a_bu(x_sn, x_pd, x_hg, a33_vp, a33_pd, a33_hg)
        a34_bu = a_bu(x_sn, x_pd, x_hg, a34_vp, a34_pd, a34_hg)
        a36_bu = a_bu(x_sn, x_pd, x_hg, a36_vp, a36_pd, a36_hg)
        a33_sb = a33_sl/a33_bu
        a34_sb = a34_sl/a34_bu
        a36_sb = a36_sl/a36_bu
    
        for m in range (0,100,1):
            x_sl = m/100.
            x_bu = 1.-x_sl
            if a33_sb < 1.:
                a33_sb_ = 1.-x_sl*(1.-a33_sb)
            else:
                a33_sb_ = 1.+x_sl*(a33_sb-1.)
            if a34_sb < 1.:
                a34_sb_ = 1.-x_sl*(1.-a34_sb)
            else:
                a34_sb_ = 1.+x_sl*(a34_sb-1.)
            if a36_sb < 1.:
                a36_sb_ = 1.-x_sl*(1.-a36_sb)
            else:
                a36_sb_ = 1.+x_sl*(a36_sb-1.)
            D33_sb_, D34_sb_, D36_sb_ = (1.-a33_sb_)*1000.,(1.-a34_sb_)*1000.,(1.-a36_sb_)*1000.
            results1 = pd.DataFrame([[x_sl,x_bu,x_sn,x_pd,x_hg,a33_sb_,a34_sb_,a36_sb_,
                                 D33_sb_,D34_sb_,D36_sb_]])
            results = pd.concat([results, results1], ignore_index=True)
            results.to_csv('a_prop_sl_bu.csv', index=False, header=False)
            
#################################
###  core mantle calculations ###
#################################

def ratio2ototal(ratio):
    ototal = ratio/(ratio + 1.)
    return ototal

def ratio_core2mantle(x,a_cf,d_i,_TS_T): # 34S/32S of mantle and core given mantle-core ratio
    alpha = a_cf
    R_i = d2R(d_i,34,VCDT) 
    H2T_i = ratio2ototal(R_i) # 34S/ST molar ratio of initial Io
    L2T_i = 1.-H2T_i # 32S/ST molar ratio of initial Io
    _32S_T = _TS_T*L2T_i
    _34S_T = _TS_T*H2T_i
    _TS_m = x*_TS_T # x is the fraction of S in the mantle

    a = alpha - 1.
    b = alpha*_TS_T + (1.-alpha)*(_32S_T+_TS_m)
    c = -_TS_m*_32S_T
    
    _32S_m = (-b + ((b**2 - (4*a*c))**0.5))/(2.*a)
    _34S_m = _TS_m - _32S_m
    R_mantle = _34S_m/_32S_m # R mantle
    _32S_c = _32S_T - _32S_m
    _34S_c = _34S_T - _34S_m
    R_core = _34S_c/_32S_c # R core
    return R_mantle, R_core

def calc_iso_variable(d_i,step_size): # isotopic composition for variable S fractions between core and mantle
    
    # set up results table
    results = pd.DataFrame([["Alpha core formation","Initial Io d34S"]])
    results1 = pd.DataFrame([[a34_cf,d_i]])
    results = pd.concat([results, results1], ignore_index=True)
    results1 = pd.DataFrame([["Fraction of S in mantle","Fraction of S in core","R34 mantle","d34S mantle","R34 core","d34S core"]])
    results = pd.concat([results, results1], ignore_index=True)
    results1 = pd.DataFrame([[0,1,"","",d2R(d_i,34,VCDT),d_i]])
    results = pd.concat([results, results1], ignore_index=True)
    results.to_csv('iso_variable_core_formation.csv', index=False, header=False)

    start = 1
    end = round(1/step_size)
    _TS_T = 1.
    
    for s in range(start,end,1): # steps
    
        x = s*step_size
        R_m, R_c = ratio_core2mantle(x,a34_cf,d_i,_TS_T)
    
        results1 = pd.DataFrame([[x,(1.-x),R_m,R2d(R_m,34,VCDT),R_c,R2d(R_c,34,VCDT)]])
        results = pd.concat([results, results1], ignore_index=True)
        results.to_csv('iso_variable_core_formation.csv', index=False, header=False)