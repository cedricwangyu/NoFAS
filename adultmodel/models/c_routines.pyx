import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp,fmod,M_PI,cos,fabs,isnan
from libc.stdio cimport printf
from libcpp cimport bool

# mmHg to Baryes Conversion Factor
cdef double mmHgToBarye = 1333.22
cdef double baryeTommHg = 1.0/1333.22

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef c_interp(double t,int n,double[:] x,double[:] y):

  cdef bool found = False
  cdef int count = 1
  while((not found) and (count<n)):
    found = (t >= x[count-1]) and (t <= x[count])
    if(not(found)):
      count += 1
  if(not(found)):
    printf("ERROR: value not found in c_interp.\n")
    exit(-1)
  else:
    return y[count-1] + (t-x[count-1])*(y[count]-y[count-1])/(x[count]-x[count-1])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef evalDeriv_adult(double t,double[::1] y,double[::1] params,
                    int numState,int numAuxState,debugMode=False):

  cdef double[:] params_view = params
  cdef double[:] y_view = y

  # Assign Parameters
  cdef double HR = params_view[0] # Heart Rate
  # Atrial and ventricular duration and shift
  cdef double tsas = params_view[1] # Atrial relative activation duration
  cdef double tpws = params_view[2] # Atrial relative activation time shift
  cdef double tsvs = params_view[3] # Ventricular relative activation duration
  # Atrial Model Parameters
  cdef double K_pas_ra_1 = params_view[4]
  cdef double K_pas_ra_2 = params_view[5]
  cdef double Emax_ra    = params_view[6]
  cdef double Vra0       = params_view[7]
  cdef double K_pas_la_1 = params_view[8]
  cdef double K_pas_la_2 = params_view[9]
  cdef double Emax_la    = params_view[10]
  cdef double Vla0       = params_view[11]
  # Ventricular Model Parameters
  cdef double K_pas_rv_1 = params_view[12]
  cdef double K_pas_rv_2 = params_view[13]
  cdef double Emax_rv    = params_view[14]
  cdef double Vrv0       = params_view[15]
  cdef double K_pas_lv_1 = params_view[16]
  cdef double K_pas_lv_2 = params_view[17]
  cdef double Emax_lv    = params_view[18]
  cdef double Vlv0       = params_view[19]
  # Atrial and Ventricular Inductances and Resistances
  cdef double L_ra_rv    = params_view[20]
  cdef double R_ra_rv    = params_view[21]
  cdef double L_rv_pa    = params_view[22]
  cdef double R_rv_pa    = params_view[23]
  cdef double L_la_lv    = params_view[24]
  cdef double R_la_lv    = params_view[25]
  cdef double L_lv_ao    = params_view[26]
  cdef double R_lv_ao    = params_view[27]
  # Aortic Arch
  cdef double C_ao       = params_view[28]
  # Pulmonary Resistance and Capacitance
  cdef double C_pa       = params_view[29]
  cdef double R_pa       = params_view[30]
  # Systemic Resistance and Capacitance
  cdef double C_sys      = params_view[31]
  cdef double R_sys_a    = params_view[32]
  cdef double R_sys_v    = params_view[33]

  # Assign state variables
  cdef double V_ra    = y_view[0]
  cdef double V_la    = y_view[1]
  cdef double V_rv    = y_view[2]
  cdef double V_lv    = y_view[3]
  cdef double Q_ra_rv = y_view[4]
  cdef double P_pa    = y_view[5]
  cdef double Q_rv_pa = y_view[6]
  cdef double Q_la_lv = y_view[7]
  cdef double P_ao    = y_view[8]
  cdef double Q_lv_ao = y_view[9]
  cdef double P_sys   = y_view[10]

  cdef int loopA
  cdef int loopB
  
  # SET OPTIONS
  printMessages = False
  totalStates = len(y)
  
  # Set Time integration parameters
  cdef double tc = 60.0/HR
  cdef double tsa = tc * tsas
  cdef double tsv = tc * tsvs
  cdef double tpw = tc/float(tpws)
  cdef double tcr = fmod(t,tc)
  
  # ====================================================
  # PART I - ATRIAL ACTIVATION AND VENTRICULAR ELASTANCE
  # ====================================================
  
  # Heart function
  cdef double tmv = tcr # Ventricle time
  cdef double tma = fmod(t+tsa-tpw,tc) # Atrium time
  
  # Ventricle activation
  cdef double fAV = 0.0
  if(tmv<tsv):
    fAV = (1.0-cos(2.0*M_PI*tmv/float(tsv)))/2.0
  else:
    fAV = 0.0
  
  # Atrium activation
  cdef double fAA = 0.0
  if(tma<tsa):
    fAA = (1.0-cos(2.0*M_PI*tma/float(tsa)))/2.0
  else:
    fAA = 0.0
  
  # ATRIA
  # Compute exponential atrial passive pressures curves
  cdef double P_pas_ra = K_pas_ra_1*(exp((V_ra-Vra0)*K_pas_ra_2)-1.0)
  cdef double P_pas_la = K_pas_la_1*(exp((V_la-Vla0)*K_pas_la_2)-1.0)
  # Compute linear atrial active pressure curves
  cdef double P_act_ra = Emax_ra*(V_ra-Vra0)
  cdef double P_act_la = Emax_la*(V_la-Vla0)
  # Blend with activation function
  cdef double P_ra = (P_pas_ra + fAA * (P_act_ra - P_pas_ra)) * mmHgToBarye
  cdef double P_la = (P_pas_la + fAA * (P_act_la - P_pas_la)) * mmHgToBarye
  
  # VENTRICLES
  # Passive Curve - Exponential
  cdef double P_pas_rv = K_pas_rv_1*(exp((V_rv-Vrv0)*K_pas_rv_2)-1.0)
  cdef double P_pas_lv = K_pas_lv_1*(exp((V_lv-Vlv0)*K_pas_lv_2)-1.0)
  # Active Curve - Linear
  cdef double P_act_rv = Emax_rv*(V_rv-Vrv0)
  cdef double P_act_lv = Emax_lv*(V_lv-Vlv0)
  # Use Activation to blend between active and passive Curves
  cdef double P_rv = (P_pas_rv + fAV * (P_act_rv - P_pas_rv)) * mmHgToBarye
  cdef double P_lv = (P_pas_lv + fAV * (P_act_lv - P_pas_lv)) * mmHgToBarye
  
  # ========================
  # PART II - HEART CHAMBERS
  # ========================
  
  # Initialize variables for valves
  # 1.0 - Valve Open; 0.0 - Valve Closed
  cdef np.ndarray[np.double_t, ndim=1] Ind = np.ones(totalStates,dtype=np.double)
  
  # Check if RA-RV Valve is open
  if( (P_rv >= P_ra) or (Q_ra_rv < 0.0) ):
    Ind[4]=0.0
  
  # Right Atrium Equation
  cdef double Q_ra_rv_p = 0.0
  if( Ind[4] > 0.0 ):
    Q_ra_rv_p = (1.0/L_ra_rv)*(P_ra - P_rv - R_ra_rv * Q_ra_rv);

  if(debugMode):
    printf("Q_ra_rv_p: %f\n", Q_ra_rv_p)
  
  # Check if RV-PA Valve is open
  if( (P_pa >= P_rv) or (Q_rv_pa < 0.0) ):
    Ind[6]=0.0
  
  # Right Ventricle
  cdef double Q_rv_pa_p = 0.0
  if( Ind[6] > 0.0 ):
    Q_rv_pa_p = (1.0/L_rv_pa)*(P_rv - P_pa - R_rv_pa * Q_rv_pa)

  if(debugMode):
    printf("Q_rv_pa_p: %f\n", Q_rv_pa_p)
  
  # Pulmonary Circulation: Only Capacitance Equation
  cdef double Q_pul = (P_pa - P_la)/R_pa
  cdef double P_pa_p = ( 1.0 / C_pa )*( Q_rv_pa * Ind[6] - Q_pul )
  
  if(debugMode):
    printf("Q_pul: %f\n", Q_pul)
    printf("P_pa_p: %f\n", P_pa_p)
  
  # Check if LA-LV Valve is open
  if( (P_lv >= P_la) or (Q_la_lv < 0.0) ):
    Ind[7]=0.0
  
  # Left Atrium
  cdef double Q_la_lv_p = 0.0
  if( Ind[7] > 0.0 ):
    Q_la_lv_p = (1.0/L_la_lv)*( P_la - P_lv - R_la_lv * Q_la_lv )

  if(debugMode):
    printf("Q_la_lv_p: %f\n", Q_la_lv_p)
  
  # Check if LV-AO Valve is open
  if( (P_ao >= P_lv) or (Q_lv_ao < 0.0) ):
    Ind[9]=0.0
  
  # Left Ventricle
  cdef double Q_lv_ao_p = 0.0
  if( Ind[9] > 0.0 ):
    Q_lv_ao_p = (1.0/L_lv_ao)*(P_lv - P_ao - R_lv_ao * Q_lv_ao)

  if(debugMode):
    printf("Q_lv_ao_p: %f\n", Q_lv_ao_p)
  
  # Flow in VEINS
  cdef double Q_sys_a = ( P_ao - P_sys ) / R_sys_a
  if(debugMode):
    printf("Q_sys_a: %f\n", Q_sys_a)
  
  # ====================================================
  # COMPUTE THE VARIATION IN VOLUME FOR ALL THE CHAMBERS
  # ====================================================
      
  # Add Systemic Resistance
  cdef double Q_sys_v = ( P_sys - P_ra ) / R_sys_v
  
  # Variation of Volume in Atria and Ventricles
  cdef double V_ra_p = Q_sys_v - Q_ra_rv * Ind[4]
  cdef double V_rv_p = Q_ra_rv * Ind[4] - Q_rv_pa * Ind[6]
  cdef double V_la_p = Q_pul - Q_la_lv * Ind[7]
  cdef double V_lv_p = Q_la_lv * Ind[7] - Q_lv_ao * Ind[9]
  
  if(debugMode):
    printf("Q_sys_v: %f\n",Q_sys_v)
    printf("V_ra_p: %f\n",V_ra_p)
    printf("V_la_p: %f\n",V_la_p)
    printf("V_rv_p: %f\n",V_rv_p)
    printf("V_lv_p: %f\n",V_lv_p)
  
  # ======================
  # PART III - AORTIC ARCH
  # ======================
  
  # Aortic arch capacitance
  cdef double P_ao_p = (1.0/C_ao) * ( Q_lv_ao - Q_sys_a )
  
  if(debugMode):
    printf("P_ao_p: %f\n",P_ao_p)
  
  # Systemic Capacitance
  cdef double P_sys_p = (1.0/C_sys) * ( Q_sys_a - Q_sys_v )
  
  # Store the derivatives
  cdef np.ndarray[np.double_t, ndim=1] res = np.zeros(2*numState + numAuxState)
  res[0]  = V_ra_p
  res[1]  = V_la_p
  res[2]  = V_rv_p
  res[3]  = V_lv_p
  res[4]  = Q_ra_rv_p
  res[5]  = P_pa_p
  res[6]  = Q_rv_pa_p
  res[7]  = Q_la_lv_p
  res[8]  = P_ao_p
  res[9]  = Q_lv_ao_p
  res[10] = P_sys_p
  res[11] = t # Current Time
  res[12] = tcr # Relative Cycle Time
  res[13] = fAA # Atrial Activation Function
  res[14] = fAV # Right Ventricle Elastance
  res[15] = 0.0
  res[16] = P_ra # Right Atrial Pressure
  res[17] = P_la # Left Atrial Pressure
  res[18] = P_rv # Right Ventricular Pressure
  res[19] = P_lv # Left Ventricular Pressure
  res[20] = Q_pul # Pulmonary flow rate
  res[21] = Q_sys_a # Systemic flow Rate - Arteries
  res[22] = Q_sys_v # Systemic flow Rate - Veins
  res[23] = Ind[4] # RA-RV valve status
  res[24] = Ind[6] # RV-PA valve status
  res[25] = Ind[7] # LA-LV valve status
  res[26] = Ind[9] # LV-AO valve status
  for loopA in range(numState):
    res[27 + loopA] = Ind[loopA]
  
  return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef evalDeriv_rc(double t,double[::1] y,double[::1] params, 
                   double cycleTime,double[:,::1] forcing,
                   int numState,int numAuxState):

  cdef double R  = params[0]
  cdef double C  = params[1]
  cdef double Pd = 55.0*mmHgToBarye
  cdef double P1 = y[0]

  # Interpolate forcing
  cdef double Q1    = c_interp(t % cycleTime, len(forcing), forcing[:,0], forcing[:,1])
  cdef double Q2    = (P1-Pd) / R
  cdef double dP1dt = (Q1-Q2) / C

  # Store the derivatives
  cdef np.ndarray[np.double_t, ndim=1] res = np.zeros(2*numState+numAuxState)
  res[0] = dP1dt
  res[1] = t # Current time
  res[2] = Pd # Fixed distal pressure at 55 mmHg
  res[3] = Q1 # Incoming flow
  res[4] = Q2 # Flow in the resistance
  res[5] = 1.0 # No valves therefore always open (1.0)

  return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def evalDeriv_rcr(double t,double[::1] y,double[::1] params,
                  double cycleTime,double[:,::1] forcing,
                  int numState,int numAuxState):

  cdef double R1 = params[0]
  cdef double R2 = params[1]
  cdef double C  = params[2]
  cdef double Pd = 55.0*mmHgToBarye

  cdef double P1 = y[0]

  # Compute other variables
  cdef double Q1    = c_interp(t % cycleTime, len(forcing), forcing[:,0], forcing[:,1])
  cdef double P0    = P1 + R1*Q1
  cdef double Q2    = (P1 - Pd) / R2
  cdef double dP1dt = (Q1 - Q2) / C

  # Store the derivatives
  cdef np.ndarray[np.double_t, ndim=1] res = np.zeros(2*numState+numAuxState)
  res[0] = dP1dt
  res[1] = t
  res[2] = Pd
  res[3] = P0
  res[4] = Q1
  res[5] = Q2
  res[6] = 1.0

  return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef solveRK4_aux(evD,double timeStep,int totalSteps,int saveEvery,double[::1] y0,double[::1] params,
                   int numState,int numAuxState):

    # Intialize Current Time
    cdef double currTime = 0.0

    # Get Total number of States
    cdef int totalStates  = numState
    cdef int totAuxStates = numAuxState

    # Total Saved Steps
    cdef int totalSaved = int(totalSteps/saveEvery)

    cdef int saveIdx = 0

    cdef double[:] y0_view = y0

    # Initialize the outputs
    cdef np.ndarray[np.double_t, ndim=1] t = np.zeros(totalSaved,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] outVals = np.zeros((totalStates,totalSaved),dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] auxOutVals = np.zeros((totAuxStates,totalSaved),dtype=np.double)

    # Declare Arrays in Cython
    cdef np.ndarray[np.double_t, ndim=1] Xn  = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] Xn1 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] Xk2 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] Xk3 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] Xk4 = np.zeros(totalStates,dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=1] Ind = np.zeros(totalStates,dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=1] k1 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k2 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k3 = np.zeros(totalStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k4 = np.zeros(totalStates,dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=1] k1AuxOut = np.zeros(totAuxStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k2AuxOut = np.zeros(totAuxStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k3AuxOut = np.zeros(totAuxStates,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] k4AuxOut = np.zeros(totAuxStates,dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(2*totalStates+totAuxStates,dtype=np.double)

    # Set Initial Conditions
    cdef int loopA
    for loopA in range(totalStates):
      Xn[loopA] = y0_view[loopA]

    # Time loop    
    cdef int loopB
    for loopA in range(totalSteps):

      # Eval K1
      out[:] = evD(currTime,Xn)
      for loopB in range(totalStates):
        k1[loopB] = out[loopB] 
        Ind[loopB] = out[totalStates+totAuxStates+loopB]
      for loopB in range(totAuxStates):
        k1AuxOut[loopB] = out[totalStates+loopB] 
      
      # Eval K2
      for loopB in range(totalStates):
        Xk2[loopB] = Xn[loopB] + ((1.0/3.0)*timeStep) * k1[loopB]
      out[:] = evD(currTime + (1.0/3.0) * timeStep,Xk2)
      for loopB in range(totalStates):
        k2[loopB] = out[loopB] 
        Ind[loopB] = out[totalStates+totAuxStates+loopB]
      for loopB in range(totAuxStates):
        k2AuxOut[loopB] = out[totalStates+loopB] 

      # Eval K3
      for loopB in range(totalStates):
        Xk3[loopB] = Xn[loopB] - (1.0/3.0)*timeStep * k1[loopB] + (1.0*timeStep) * k2[loopB]
      out[:] = evD(currTime + (2.0/3.0) * timeStep,Xk3)
      for loopB in range(totalStates):
        k3[loopB] = out[loopB] 
        Ind[loopB] = out[totalStates+totAuxStates+loopB]
      for loopB in range(totAuxStates):
        k3AuxOut[loopB] = out[totalStates+loopB] 

      # Eval K4
      for loopB in range(totalStates):
        Xk4[loopB] = Xn[loopB] + timeStep*k1[loopB] - timeStep*k2[loopB] + timeStep * k3[loopB]
      out[:] = evD(currTime + timeStep,Xk4)
      for loopB in range(totalStates):
        k4[loopB] = out[loopB] 
        Ind[loopB] = out[totalStates+totAuxStates+loopB]
      for loopB in range(totAuxStates):
        k4AuxOut[loopB] = out[totalStates+loopB] 

      # Eval Xn1 Update      
      for loopB in range(totalStates):
        Xn1[loopB] = Xn[loopB] + (1.0/8.0)*timeStep * (k1[loopB] + 3.0 * k2[loopB] + 3.0 * k3[loopB] + k4[loopB])
        if(Ind[loopB] <= 0):
          Xn1[loopB] = 0.0

      foundNan = False
      for loopB in range(totalStates):
        foundNan = foundNan or isnan(Xn1[loopB])
      if(foundNan):
        print("ERROR: Nan found in solution components")
        exit(-1)
   
      # If this is a "Save" step then append to solution vectors
      if(loopA % saveEvery == 0):
        # Copy Auxiliary outputs at every time step
        saveIdx = int(loopA/saveEvery)
        t[saveIdx] = currTime
        for loopB in range(totalStates):
          outVals[loopB,saveIdx] = Xn1[loopB]
        for loopB in range(totAuxStates):
          auxOutVals[loopB,saveIdx] = k4AuxOut[loopB]

      # Update Xn
      for loopB in range(totalStates):
        Xn[loopB] = Xn1[loopB]

      # Update Current Time
      currTime += timeStep

    # RETURN OK
    return t,outVals,auxOutVals

