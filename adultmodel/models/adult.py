import numpy as np
import scipy.signal as sp
from circuitModels import circuitModel,mmHgToBarye,baryeTommHg
from c_routines import evalDeriv_adult

def getMean(t,y):
  '''
  Get the mean of a signal in time
  Computing the integral and dividing by the total time
  '''
  return np.trapz(y,t)/(t[-1]-t[0])

def getMeanValveOpen(t,y):
  newY = y.copy()
  newY[newY > 0] = 1.0
  timePositive = np.trapz(newY,t)
  return np.trapz(y,t)/timePositive

def zeroAtValveOpening(curve,valveIsOpen):
  """
  Determines valve opening time and cyclically shifts curve
  """
  if(np.count_nonzero(valveIsOpen) > 0):
    if(valveIsOpen[0] == 0):
      valveOpeningIDX = valveIsOpen.nonzero()[0][0]
    else:
      rev = np.asarray(np.logical_xor(valveIsOpen,np.ones(len(valveIsOpen))))
      valveOpeningIDX = np.nonzero(rev)[0][-1]
    # Return circular shif of vector starting from valve opening
    return np.roll(curve,-valveOpeningIDX)
  else:
    return None

class peakRecord(object):
  def __init__(self,iMin,tMin,yMin,iMax,tMax,yMax):
    self.iMin = iMin
    self.tMin = tMin
    self.yMin = yMin
    self.iMax = iMax
    self.tMax = tMax
    self.yMax = yMax

def getPeaks(t,curve):

  # Find Peaks
  idxMax = sp.find_peaks(curve)[0]
  idxMin = sp.find_peaks(-curve)[0]

  # Store Peaks in record and return
  peaks = peakRecord(idxMin,t[idxMin],curve[idxMin],idxMax,t[idxMax],curve[idxMax])
  return peaks

def getAccelerationTime(peaks,t0):
  if(len(peaks.iMax) > 0):
    # The first maximum is assumed to be S
    at = (peaks.tMax[0]-t0) * 1000.0;
    return True,at
  else:
    return False,0.0

def getDecelerationTime(peaks):
  if( (len(peaks.iMax) > 0) and (len(peaks.iMin) > 0)):
    # The minimum (M) must follow the maximum (S)
    if(peaks.tMin[0] > peaks.tMax[0]):      
      dt = (peaks.tMin[0] - peaks.tMax[0]) * 1000.0
      return True,dt
    else:
      return False,0.0
  else:
    return False,0.0

def getEARatio(peaks):
  # At least two maxima
  if(len(peaks.iMax) > 1):
    EARatio = (peaks.yMax[0]/peaks.yMax[1])
    return True,EARatio
  else:
    return False,0.0

class lpnAdultModel(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None,debugMode=False):
    # Init parameters
    numParam    = 45
    numState    = 11
    numAuxState = 16
    numOutputs  = 28
    self.debugMode = debugMode

    icName = np.array([# Initial Volumes
                       "V_ra_ini","V_la_ini","V_rv_ini","V_lv_ini",
                       # Initial Pressurs and Flows
                       "Q_ra_rv_ini","P_pa_ini","Q_rv_pa_ini","Q_la_lv_ini","P_ao_ini","Q_lv_ao_ini","P_sys_ini"])

    parName = np.array([# Heart Timing Parameters 
                        "HR","tsas","tpws","tsvs",
                        # Right Atrial Parameters
                        "K_pas_ra_1","K_pas_ra_2","Emax_ra","Vra0",
                        # Left Atrial Parameters
                        "K_pas_la_1","K_pas_la_2","Emax_la","Vla0",
                        # Right Ventricular Parameters
                        "K_pas_rv_1","K_pas_rv_2","Emax_rv","Vrv0",
                        # Left Ventricular Parameters 
                        "K_pas_lv_1","K_pas_lv_2","Emax_lv","Vlv0",
                        # Valve Resistance and Inductance
                        "L_ra_rv","R_ra_rv","L_rv_pa","R_rv_pa",
                        "L_la_lv","R_la_lv","L_lv_ao","R_lv_ao",
                        # Aortic, Pulmonary and Systemic Resistance/Compliance
                        "C_ao","C_pa","R_pa","C_sys","R_sys_a","R_sys_v"])

    resName = np.array(["heart_rate2", # heartRate - ip_0002_heart_rate2
                        "systolic_bp_2", # maxAOPress - ip_0002_systolic_bp_2
                        "diastolic_bp_2", # minAOPress - ip_0002_diastolic_bp_2
                        "cardiac_output", # CO - ip_0002_cardiac_output
                        "systemic_vascular_resistan", # params[32]+params[33] - ip_0002_systemic_vascular_resistan
                        "pulmonary_vascular_resista", # params[30] - ip_0002_pulmonary_vascular_resista
                        "cvp", # avRAPress - ip_0002_cvp
                        "right_ventricle_diastole", # minRVPress - ip_0002_right_ventricle_diastole
                        "right_ventricle_systole", # maxRVPress - ip_0002_right_ventricle_systole
                        "left_ventricle_diastole", # minLVPress
                        "left_ventricle_systole", # maxLVPress
                        "rvedp", # RVEDP - ip_0002_rvedp
                        "aov_mean_pg", # meanAOVPG - ip_0002_aov_mean_pg
                        "aov_peak_pg", # maxAOVPG - ip_0002_aov_peak_pg
                        "mv_decel_time", # mvDecelTime - ip_0002_mv_decel_time
                        "mv_e_a_ratio", # mvEARatio - ip_0002_mv_e_a_ratio
                        "pv_at", # pvAccelTime - ip_0002_pv_at
                        "pv_max_pg", # maxPVPG - ip_0002_pv_max_pg
                        "ra_pressure", # avRAPress - ip_0002_ra_pressure
                        "ra_vol_a4c", # minRAVolume - ip_0002_ra_vol_a4c - End Systolic
                        "la_vol_a4c", # minLAVolume - ip_0002_la_vol_a4c - End Systolic
                        "lv_esv", # minLVVolume - ip_0002_lv_esv
                        "lv_vol_a4c", # maxLVVolume - ip_0002_lv_vol
                        "lvef", # LVEF - ip_0002_lvef
                        "pap_diastolic", # minPAPress - ip_0002_pap_diastolic
                        "pap_systolic", # maxPAPress - ip_0002_pap_systolic
                        "wedge_pressure"]) # avPCWPress - ip_0002_wedge_pressure

    limits = np.array([# Heart Cycle Parameters
                       [40.0,100.0], # HR - Heart Rate
                       # Atrial and ventricular activation duration and shift
                       [0.05,0.4], # tsas - Atrial relative activation duration
                       [5.0,10.0], # tpws - Atrial relative activation time shift
                       [0.1,0.5], # tsvs - Ventricular relative activation duration
                       # Atrial Model Parameters
                       [0.1,10.0], # K_pas_ra_1 - Atrial passive curve slope, right atrium
                       [0.0001,0.06], # K_pas_ra_2 - Atrial passive curve exponent factor, right atrium
                       [0.05,5.0], # Emax_ra - Atrial active curve slope, right atrium
                       [0.0,50.0], # Vra0 - Unstressed right atrial volume
                       [0.1,10.0], # K_pas_la_1 - Atrial passive curve slope, left atrium
                       [0.0001,0.06], # K_pas_la_2 - Atrial passive curve exponent factor, left atrium
                       [0.05,5.0], # Emax_la - Atrial active curve slope, left atrium
                       [0.0,50.0], # Vla0 - Unstressed left atrial volume
                       # Ventricular Model Parameters
                       [0.1,20.0], # K_pas_rv_1 - Ventricular passive curve slope, right ventricle
                       [0.0001,0.01], # K_pas_rv_2 - Ventricular passive curve exponent factor, right ventricle
                       [0.1,5.0], # Emax_rv - Ventricular active curve slope, right ventricle
                       [0.0,50.0], # Vrv0 - Unstressed right atrial volume
                       [0.1,20.0], # K_pas_lv_1 - Ventricular passive curve slope, left ventricle
                       [0.0001,0.01], # K_pas_lv_2 - Ventricular passive curve exponent factor, left ventricle
                       [1.0,5.0], # Emax_lv - Ventricular active curve slope, left ventricle
                       [0.0,50.0], # Vlv0 - Unstressed left atrial volume
                       # Atrial and Ventricular Inductances and Resistances
                       [0.1,0.1], # L_ra_rv - Inductance of right atrium
                       [10.0,10.0], # R_ra_rv - Resistance of right atrium
                       [0.1,0.1], # L_rv_pa - Inductance of right ventricle
                       [15.0,15.0], # R_rv_pa - Resistance of right ventricle
                       [0.1,0.1], # L_la_lv - Inductance of left atrium
                       [8.0,8.0], # R_la_lv - Resistance of left atrium
                       [0.1,0.1], # L_lv_ao - Inductance of left ventricle
                       [25.0,25.0], # R_lv_ao - Resistance of left ventricle
                       # Aortic Arch
                       [1.0e-5,0.001], # C_ao - Aortic capacitance
                       # Pulmonary Resistance and Capacitance
                       [100.0e-6,0.01], # C_pa - Pulmonary capacitance
                       [1.0,500.0], # R_pa - Pulmonary resistance
                       # Systemic Resistance and Capacitance
                       [100.0e-6,0.05], # C_sys - Systemic capacitance
                       [1.0,3000.0], # R_sys_a - Systemic Resistance - Arteries
                       [1.0,3000.0]]) # R_sys_v - Systemic Resistance - Veins

    # NOTE: CGS Units: Pressures in Barye, Flowrates in mL/s
    # Default Initial Conditions
    defIC = np.array([# Initial Values
                      0.0, # V_ra
                      0.0, # V_la
                      0.0, # V_rv
                      0.0, # V_lv
                      0.0, # Q_ra_rv
                      70.0 * mmHgToBarye, # P_pa
                      0.0, # Q_rv_pa
                      0.0, # Q_la_lv
                      100.0 * mmHgToBarye, # P_ao
                      0.0, # Q_lv_ao
                      50.0 * mmHgToBarye]) # P_syss

    # NOTE: CGS Units: Pressures in Barye, Flowrates in mL/s
    defParam = np.array([# Heart Cycle Parameters
                         78.0, # HR - Heart Rate (beats per minute)         
                         # Atrial and ventricular activation duration
                         0.2, # tsas - Atrial relative activation duration
                         9.5, # tpws - Atrial relative activation time shift
                         0.4, # tsvs - Ventricular relative activation duration
                         # Atrial model parameters
                         5.0, # K_pas_ra_1 - Atrial passive curve slope, right atrium
                         0.006, # K_pas_ra_2 - Atrial passive curve exponent factor, right atrium
                         0.1, # Emax_ra - Atrial active curve slope, right atrium
                         0.0, # Vra0 - Unstressed right atrial volume
                         5.0, # K_pas_la_1 - Atrial passive curve slope, left atrium
                         0.0065, # K_pas_la_2 - Atrial passive curve exponent factor, left atrium
                         0.2, # Emax_la - Atrial active curve slope, left atrium
                         0.0, # Vla0 - Unstressed left atrial volume
                         # Ventricular Model Parameters
                         5.0, # K_pas_rv_1 - Ventricular passive curve slope, right atrium
                         0.003, # K_pas_rv_2 - Ventricular passive curve exponent factor, right atrium
                         0.5, # Emax_rv - Ventricular active curve slope, right atrium
                         0.0, # Vrv0 - Unstressed right atrial volume
                         2.0, # K_pas_lv_1 - Ventricular passive curve slope, left atrium
                         0.003, # K_pas_lv_2 - Ventricular passive curve exponent factor, left atrium
                         4.0, # Emax_lv - Ventricular active curve slope, left atrium
                         20.0, # Vlv0 - Unstressed left atrial volume
                         # Atrial and Ventricular Inductances and Resistances
                         0.1, # L_ra_rv - Inductance of right atrium
                         10.0, # R_ra_rv - Resistance of right atrium
                         0.1, # L_rv_pa - Inductance of right ventricle
                         15.0, # R_rv_pa - Resistance of right ventricle
                         0.1, # L_la_lv - Inductance of left atrium
                         8.0, # R_la_lv - Resistance of left atrium
                         0.1, # L_lv_ao - Inductance of left ventricle
                         25.0, # R_lv_ao - Resistance of left ventricle
                         # Aortic Arch
                         1000.0e-6, # C_ao - Aortic capacitance
                         # Pulmonary Resistance and Capacitance
                         4000.0e-6, # C_pa - Pulmonary capacitance
                         130.0, # R_pa - Pulmonary resistance
                         # Systemic Resistance and Capacitance
                         400.0e-6, # C_sys - Systemic Capacitance
                         400.0, # R_sys_a - Systemic Resistance - Arteries
                         1200.0]) # R_sys_v - Systemic Resistance - Veins

    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     icName,parName,resName,
                     limits,defIC,defParam,
                     cycleTime,totalCycles)

  def evalDeriv(self,t,y,params):
    return evalDeriv_adult(t,y,params,self.numState,self.numAuxState)

  def postProcess(self,t,y,aux,start,stop):

    # HEART RATE PARAMETER
    heartRate = self.params[0]

    # SYSTOLIC, DIASTOLIC AND AVERAGE BLOOD PRESSURES
    minAOPress = np.min(y[8,start:stop])
    maxAOPress = np.max(y[8,start:stop])
    avAOPress  = getMean(t[start:stop],y[8,start:stop])
    
    # RA PRESSURE
    minRAPress = np.min(aux[5,start:stop])
    maxRAPress = np.max(aux[5,start:stop])
    avRAPress  = getMean(t[start:stop],aux[5,start:stop])
    
    # RV PRESSURE
    minRVPress  = np.min(aux[7,start:stop])
    maxRVPress  = np.max(aux[7,start:stop])
    avRVPress   = getMean(t[start:stop],aux[7,start:stop])
        
    # SYSTOLIC, DIASTOLIC AND AVERAGE PA PRESSURES
    minPAPress = np.min(y[5,start:stop])
    maxPAPress = np.max(y[5,start:stop])
    avPAPress  = getMean(t[start:stop],y[5,start:stop])
    
    # PWD OR AVERAGE LEFT ATRIAL PRESSURE
    # AVERAGE PCWP PRESSURE - INDIRECT MEASURE OF LEFT ATRIAL PRESSURE
    avPCWPress = getMean(t[start:stop],aux[6,start:stop])
    
    # LEFT VENTRICULAR PRESSURES
    minLVPress  = np.min(aux[8,start:stop])
    maxLVPress  = np.max(aux[8,start:stop])
    avgLVPress  = getMean(t[start:stop],aux[8,start:stop])
        
    # CARDIAC OUTPUT
    CO = getMean(t[start:stop],y[9,start:stop])
    
    # LEFT AND RIGHT VENTRICULAR VOLUMES
    minRVVolume = np.min(y[2,start:stop])
    maxRVVolume = np.max(y[2,start:stop])
    minLVVolume = np.min(y[3,start:stop])
    maxLVVolume = np.max(y[3,start:stop])

    # END SYSTOLIC RIGHT ATRIAL VOLUME
    minRAVolume = np.min(y[0,start:stop])
    
    # END SYSTOLIC LEFT ATRIAL VOLUME
    minLAVolume = np.min(y[1,start:stop])
    
    # EJECTION FRACTION
    LVEF = ((maxLVVolume - minLVVolume)/maxLVVolume)*100.0
    RVEF = ((maxRVVolume - minRVVolume)/maxRVVolume)*100.0
    
    # RIGHT VENTRICULAR VOLUME AT BEGINNING OF SYSTOLE
    RVEDP = aux[7,start]
    
    # PRESSURE GRADIENT ACROSS AORTIC VALVE
    output = np.empty(stop-start)
    output[:] = np.abs(y[8,start:stop] - aux[8,start:stop]) * aux[15,start:stop] # fabs(aortic - LV) * IND(AOV)
    maxAOVPG  = np.max(output)
    meanAOVPG = getMeanValveOpen(t[start:stop],output)
    
    # PRESSURE GRADIENT ACROSS PULMONARY VALVE
    output[:] = np.abs(y[5,start:stop] - aux[7,start:stop]) * aux[13,start:stop] # fabs(pulmonary - RV) * IND(PV)
    maxPVPG  = np.max(output)
    meanPVPG = getMeanValveOpen(t[start:stop],output)
    
    # MITRAL FLOW - REPOSITION TO VALVE OPENING
    mvflowFromValveOpening = zeroAtValveOpening(y[7][start:stop],aux[14][start:stop])
    if(mvflowFromValveOpening is None):
      print("ERROR: Mitral valve is not opening in last heart cycle.")
    # FIND MITRAL FLOW PEAKS
    mitralPeaks = getPeaks(t[start:stop],mvflowFromValveOpening)
    # MITRAL VALVE DECELERATION TIME
    isDecelTimeOK,mvDecelTime = getDecelerationTime(mitralPeaks)
    # MITRAL VALVE E/A RATIO
    isMVEARatioOK,mvEARatio = getEARatio(mitralPeaks)

    if(False):
      plt.plot(t[start:stop],mvflowFromValveOpening)
      plt.scatter(mitralPeaks.tMax,mitralPeaks.yMax)
      plt.scatter(mitralPeaks.tMin,mitralPeaks.yMin)
      plt.show()
      exit(-1)

    # PULMONARY VALVE ACCELERATION TIME
    # SHIFT CURVE WITH BEGINNING AT VALVE OPENING
    pvflowFromValveOpening = zeroAtValveOpening(y[6][start:stop],aux[13][start:stop])
    if(pvflowFromValveOpening is None):
      print("ERROR: Second Valve is not opening in heart cycle.")
      exit(-1)
    # FIND PULMONARY FLOW PEAKS
    pulmonaryPeaks = getPeaks(t[start:stop],pvflowFromValveOpening)
    isPVAccelTimeOK,pvAccelTime = getAccelerationTime(pulmonaryPeaks,t[start])

    if(False):
      plt.plot(t[start:stop],pvflowFromValveOpening)
      plt.scatter(pulmonaryPeaks.tMax,pulmonaryPeaks.yMax)
      plt.scatter(pulmonaryPeaks.tMin,pulmonaryPeaks.yMin)
      plt.show()
      exit(-1)

    if(self.debugMode):
      print("mvDecelTime: %f\n",mvDecelTime);
      print("mvEARatio: %f\n",mvEARatio);
      print("pvAccelTime: %f\n",pvAccelTime);

    # ALTERNATIVE COMPUTATION OF SVR and PVR
    altSVR = 80.0*(((avAOPress/1333.22) - (avRAPress/1333.22))/(CO*(60.0/1000.0)))
    altPVR = 80.0*(((avPAPress/1333.22) - (avPCWPress/1333.22))/(CO*(60.0/1000.0)))

    # Assign Results Based on Model Version
    res = np.array([heartRate, # ip_0002_heart_rate2
                    maxAOPress * baryeTommHg, # ip_0002_systolic_bp_2
                    minAOPress * baryeTommHg, # ip_0002_diastolic_bp_2
                    CO * 60.0/1000.0, # ip_0002_cardiac_output
                    self.params[32]+self.params[33], # ip_0002_systemic_vascular_resistan
                    self.params[30], # ip_0002_pulmonary_vascular_resista
                    avRAPress * baryeTommHg, # ip_0002_cvp
                    minRVPress * baryeTommHg, # ip_0002_right_ventricle_diastole
                    maxRVPress * baryeTommHg, # ip_0002_right_ventricle_systole
                    minLVPress * baryeTommHg, # left_ventricle_diastole
                    maxLVPress * baryeTommHg, # left_ventricle_systole
                    # Right ventricular volume at beginning of systole
                    RVEDP * baryeTommHg, # ip_0002_rvedp
                    meanAOVPG * baryeTommHg, # ip_0002_aov_mean_pg
                    maxAOVPG * baryeTommHg, # ip_0002_aov_peak_pg
                    mvDecelTime, # ip_0002_mv_decel_time
                    mvEARatio, # ip_0002_mv_e_a_ratio
                    pvAccelTime, # ip_0002_pv_at
                    maxPVPG * baryeTommHg, # ip_0002_pv_max_pg
                    avRAPress * baryeTommHg, # ip_0002_ra_pressure
                    # Assume maximum (diastolic) volume
                    minRAVolume, # ip_0002_ra_vol_a4c - End Systolic
                    minLAVolume, # ip_0002_la_vol_a4c - End Systolic
                    minLVVolume, # ip_0002_lv_esv
                    # Assume maximum (diastolic) volume
                    maxLVVolume, # ip_0002_lv_vol
                    LVEF, # ip_0002_lvef
                    minPAPress * baryeTommHg, # ip_0002_pap_diastolic
                    maxPAPress * baryeTommHg, # ip_0002_pap_systolic
                    avPCWPress* baryeTommHg]) # ip_0002_wedge_pressure

    return res

# TESTING MODEL
if __name__ == "__main__":

  cycleTime = 1.07
  totalCycles = 10
  model = lpnAdultModel(cycleTime,totalCycles)

  # Get Default Initial Conditions
  y0        = model.defIC
  params    = model.defParam
  outs      = model.solve(params=params,y0=y0)
  outLabels = model.resName

  if(False):
    for loopA in range(len(outLabels)):
      print("%30s %-8f" % (outLabels[loopA],outs[loopA]))

  # Array with measurement standard deviations - same size of the model result vector
  stds = np.array([3.0,  # ip_0002_heart_rate2
                   1.5,  # ip_0002_systolic_bp_2
                   1.5,  # ip_0002_diastolic_bp_2
                   0.2,  # ip_0002_cardiac_output
                   50.0, # ip_0002_systemic_vascular_resistan
                   5.0,  # ip_0002_pulmonary_vascular_resista
                   0.5,  # ip_0002_cvp
                   1.0,  # ip_0002_right_ventricle_diastole
                   1.0,  # ip_0002_right_ventricle_systole
                   1.0,  # left_ventricle_diastole
                   1.0,  # left_ventricle_systole
                   1.0,  # ip_0002_rvedp
                   0.5,  # ip_0002_aov_mean_pg
                   0.5,  # ip_0002_aov_peak_pg
                   6.0,  # ip_0002_mv_decel_time
                   0.2,  # ip_0002_mv_e_a_ratio
                   6.0,  # ip_0002_pv_at
                   0.5,  # ip_0002_pv_max_pg
                   0.5,  # ip_0002_ra_pressure
                   3.0,  # ip_0002_ra_vol_a4c - End Systolic
                   3.0,  # ip_0002_la_vol_a4c - End Systolic
                   10.0, # ip_0002_lv_esv
                   20.0, # ip_0002_lv_vol
                   2.0,  # ip_0002_lvef
                   1.0,  # ip_0002_pap_diastolic
                   1.0,  # ip_0002_pap_systolic
                   1.0]) # ip_0002_wedge_pressure

  # Evaluate Model Log-Likelihood
  # dbFile = '../data/EHR_dataset.csv'
  dbFile = '../data/validation_dataset.csv'
  columnID = 0 # First Patient

  ll = model.evalNegLL(columnID,dbFile,stds,params,y0)

  print("Model Negative LL: ",ll)
