import numpy as np
from circuitModels import circuitModel,mmHgToBarye,baryeTommHg
from c_routines import evalDeriv_rc

class rcModel(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None):
    # Init parameters
    numParam    = 2
    numState    = 1
    numAuxState = 4
    numOutputs  = 3
    parName  = np.array(["R","C"])
    icName   = np.array(["p_ini"])
    resName  = np.array(["min_pressure",
                        "max_pressure",
                        "avg_pressure"])
    limits   = np.array([[100.0, 1500.0],
                       [1.0e-5, 1.0e-2]])
    defIC    = np.array([55.0*mmHgToBarye])
    defParam = np.array([1000.0,0.00005])    
    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     icName,parName,resName,
                     limits,defIC,defParam,
                     cycleTime,totalCycles,forcing=forcing)

  def evalDeriv(self,t,y,params):
    return evalDeriv_rc(t,y,params,
                        self.cycleTime,self.forcing,
                        self.numState,self.numAuxState)
                   
  def postProcess(self,t,y,aux,start,stop):
    res = np.zeros(self.numOutputs)
    # Compute Min Pressure
    res[0] = np.amin(y[0,start:stop])/mmHgToBarye
    # Compute Max Pressure
    res[1] = np.max(y[0,start:stop])/mmHgToBarye
    # Compute Average Pressure
    res[2] = (np.trapz(y[0,start:stop],t[start:stop])/float(t[stop-1]-t[start]))/mmHgToBarye
    return res

# TESTING MODEL
if __name__ == "__main__":

  # Create the model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('../data/inlet.flow')
  model = rcModel(cycleTime,totalCycles,forcing)

  # Get Default Initial Conditions
  y0        = model.defIC
  # Get Default Model Parameters
  params    = model.defParam
  # Solve Model and Get Outputs
  outs      = model.solve(params=params,y0=y0)[0]
  outLabels = model.resName

  # Array with measurement standard deviations - same size of the model result vector
  stds = np.array([1.0,  # min_pressure
                   1.0,  # max_pressure
                   1.0]) # avg_pressure

  # Evaluate Model Log-Likelihood
  # dbFile = '../data/EHR_dataset.csv'
  dbFile = '../data/rc_dataset.csv'
  columnID = 0 # First Patient

  ll = model.evalNegLL(columnID,dbFile,stds,params,y0)

  print("Model Negative LL: ",ll)






