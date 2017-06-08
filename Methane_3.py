#packages and libraries
import pylab
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from math import pi
from math import ceil
from math import floor
# Pre-Processing Functions
def timeAlign(STR, lagTime, j):
  # Adjusts time series file to account for sampling line delay
  #
  # Args: 
  # STR : time series data.frame in standard format
  # lag : sampling line delay in same units as STR
  # 
  # Returns:
  #   time-aligned data.frame
    dat = STR.sub((lagTime))
    #dat[1:dat_nrows, 'CH4.ppm'] = STR[(lagTime+1):nrows, 'CH4.ppm']
    dat["CH4ppm"]= STR[[j]].add((lagTime))
       
    return(dat);
def rotateSonic(dat,pi):
  # Rotates sonic anemometer output to streamlined coordinates
  #
  # Args: 
  # dat : time series data.frame in standard format, must have columes u, v, w
  # 
  # Returns:
  #   time-series data.frame with streamlined coordinates, 
  #   i.e. mean(u) = mean(w) = 0
  #   if mean(ws3v)>0, wd_sonic centered at 0 and ranges from -180 to 180
  #   if mean(ws3v)<0, wd_sonic centered at 180 and ranges from 0 to 360
  # Rotation angle 1
  u_mean = dat["u"].mean(skipna = True)
  v_mean = dat["v"].mean(skipna = True)
  RA = np.arctan2(-u_mean, -v_mean) + pi 
  cos_RA = np.cos(RA)
  sin_RA = np.sin(RA)
  vPrime = dat["v"]*cos_RA + dat["u"]*sin_RA
  dat["u"] = (-1)*dat["v"]*sin_RA + dat["u"]*cos_RA
     
  # Rotation angle 2
  w_mean = dat["w"].mean(skipna = True)
  vPrime_mean = vPrime.mean(skipna = True)
  RB = np.arctan2(-w_mean, -vPrime_mean) + pi
  cos_RB = np.cos(RB)
  sin_RB = np.sin(RB)
  dat["v"] = vPrime*cos_RB + dat["w"]*sin_RB
  dat["w"] = (-1)*vPrime*sin_RB + dat["w"]*cos_RB
  
  # Calculate new 2D wind direction and speed
  dat["wd_sonic"] = np.arctan2(dat["u"],dat["v"])*180/pi
  u_squ = dat["u"]*dat["u"]
  v_squ = dat["v"]*dat["v"]
  dat["ws_sonic"] = np.sqrt(u_squ + v_squ)
  
  return(dat);
def subtractBackground(CH4,CH4back):
#  # Subtracts background concentration from methane
#  #
#  # Args: 
#  # CH4 : methane time series data.frame 
#  # 
#  # Returns:
#  #   methan time-series data.frame after background subtraction
#  #set filter at 25th percentile, only use data above this level
  CH4_25 = np.percentile(STR["CH4ppm"], 25)
  CH4_25sort = [k for k in STR["CH4ppm"] if k <= CH4_25 ]
  deltafilter = np.mean(CH4_25sort)
  for i in range(1,len(CH4)): 
      if(CH4[i]>deltafilter): 
          CH4[i] = CH4[i] - CH4back[i]
      else:
          CH4[i] = None 
  return(CH4);

def convertppmtogperm3(a1, pbar, tbar):
#  # Converts methane concentration from ppm to g/m3
#  #
#  # Args: 
#  # a1 : methane concentration
#  # pbar : mean pressure (mbar)
#  # tbar : mean temperature (K)
#  #
#  # Returns:
#  # a1 in g/m3
#  
#  # Ideal gas constant R = 8.3144598 x 10-2 m3 mbar K-1 mol-1
#  # Molecular weight of CH4 = 16.04
  a1gperm3 = a1*10**-4*pbar*16.04/(tbar*8.3144598)
  return(a1gperm3);

def convertgperstoscfh(PSGgpers):
#  # Converts methane release rate in g/s to scfh
#  #
#  # Args: 
#  # PSGgpers:  g/s ch4
#  # pbar : mean pressure (mbar)
#  # tbar : mean temperature (K)
#  #
#  # Returns:
#  # a1 in g/m3
#  
#  # Standard cubic foot represents 1.19804 moles of gas 
#  # at 21 C and 101.35 kPa 
  PSGscfh = PSGgpers/(16.04*1.19804) * 3600
  return(PSGscfh);

#PSGc Functions
def CenterWindOnPlume(STR):
  # Center wind direction so that maximum plume concentration is located at 0 
  #
  # Args: 
  # STR : full concentration time series
  #
  # Returns: 
  # vector of centered wd 
  nrow = len(STR)
  ch4_wdbin = binConc(STR, 10, 0.002*nrow)
  delta = np.max(ch4_wdbin) 
  theta = STR["wd_sonic"] - delta
# statement, true, false
  theta = np.where(theta > 180, theta - 360, (np.where(theta < -180, theta + 360, theta)))
  return(theta);  #,ch4.wdbin)) 
def binConc(dat, binWidth, binCutFilter, colName = "wd_sonic", 
            minBin = -360, maxBin = 360):
  # Bin methane concentration by wind direction
  # Bins are closed on the left and open on the right
  #
  # Args: 
  # dat : time series data.frame
  # binWidth: integer indicating size of wd bins
  # binCutFilter: remove bins with less than this many measurements
  # colName : column on which to bin concentrations
  # minBin : minimum bin value
  # maxBin : maximum bin value
  # 
  # Returns:
  #   mean methane concentration, number of measurements, and bin center
  
  bins = np.arange(minBin, maxBin, binWidth)
# Create data.frame with wind direction bin labels and centers
  bins_df = pd.DataFrame({"Bin":pd.cut(bins,bins,right = False) ,"Center":(bins + binWidth/2)})
# subset_bins = bins_df["Bin"] != None & bins_df["Center"] != None
  dat["Bin"] = pd.cut(dat[colName], bins, right = False) 
  ch4_bin1  = dat.groupby("Bin").apply(lambda x: pd.DataFrame({"ch4":np.mean(x["CH4"]),
                                                              "ws":np.mean(x["ws_sonic"]),
                                                              "n":len(x["CH4"])}, index = [0]))
# Add bin center to aggregated values and remove bins 
# with fewer points than the filter
  n = len(ch4_bin1["n"]) 
  sub1 = ch4_bin1.div(n >= binCutFilter, bins_df)
  ch4bin = ch4_bin1.merge(sub1,how = 'right')
  print(ch4bin)
  return(bins); #supposed to return ch4bin
def GetStability(dat):
  # Get Pasquill Stability Class
  # 
  # Args:
  # dat: dataframe that contains columns "w","ws_sonic", from sonic anemometer 
  #   and "wd_met" from 2D met station
  #
  # Returns:
  # Pasquill Stability Class
  
  # Get stabilibity from turbulence intensity determined using sonic anemometer
  w_std = dat["w"].std(skipna = True)
  ws_sonicMean = dat["ws_sonic"].mean(skipna = True)
  turbint = w_std/ws_sonicMean
  turbint_breaks = [0, 0.080, 0.105, 0.130, 0.155, 0.180, 0.205, 2]
  labels = np.arange(1,7,1)  #seq equivanlent
  turbStability = pd.cut(labels, turbint_breaks, turbint)
  turbStab = pd.get_dummies(turbStability)
  # Get stability from the standard deviation of the wind direction measured using 
  # the met station, use the Yamartino method
  sin_wd = np.sin(dat["wd_sonic"])*pi/180
  cos_wd = np.cos(dat["wd_sonic"])*pi/180

  sin_wdMean = sin_wd.mean(skipna = True)
  cos_wdMean = cos_wd.mean(skipna = True)
  sin_wdMeanSqu = sin_wdMean*sin_wdMean
  cos_wdMeanSqu = cos_wdMean*cos_wdMean
  epsilon = np.sqrt(1-(sin_wdMeanSqu + cos_wdMeanSqu))
  wd_met_sd = np.arcsin(epsilon)*(1 + (2/np.sqrt(3) - 1)*(epsilon*epsilon*epsilon)) * 180/pi 
  wdsd_breaks = [0, 7.50, 11.50, 15.50, 19.50, 23.50, 27.50, 100]
  sdwdStability = pd.cut(labels, wdsd_breaks, wd_met_sd)
  sdwdStab = pd.get_dummies(sdwdStability)
  # Mean stability
  StabClass = (turbStab + sdwdStab)/2 + 0.0001
  StabilityClass = round(StabClass) 
  return(StabilityClass); #needs to return StabilityClass
def GetPGsigma (psigma, distancem, stabilityClass):
  # Get constant term in PSG estimate : 2*pi*sigmay*sigmaz
  #
  # Args: 
  # psgima: Point source gaussian look up table
  # distance: distance from source to measurement
  # stabilityClass: Pasquill stability class
  #
  # Returns: 
  # sigmay,sigmaz
  distance = round(distancem,0)
  print (distancem)
  print (distance)
  print (stabilityClass)
#  pgsigmay = as.numeric(pgsigma[which(pgsigma$dist.int == distance & 
#                                         pgsigma$PGI == stabilityClass), "sigmay"])  
  pgsigma = pd.read_csv('pgsigma.csv', header = 0)
 # pgsigmay = np.where((pgsigma["dist.int"] == distance) & (pgsigma["PGI"] == stabilityClass),pgsigma["dist.int"],pgsigma["sigmay"]) 
  #pgsigmay = pgsigma.loc[lambda df: (df["dist.int"] == distance) & (df["PGI"] == stabilityClass), :]
#  pgsigmaz = as.numeric(pgsigma[which(pgsigma$dist.int == distance & 
#                                         pgsigma$PGI == stabilityClass), "sigmaz"])
  pgsigmaz_all = pgsigma["sigmaz"]  
  #PSGconst = 2*pi*pgsigmay*pgsigmaz
  print(pgsigma)
  return(distance); #return list(pgsigmay=pgsigmay, pgsigmaz=pgsigmaz)
def GetSigmaYdata (STR, distance,Test,Skip,STRfile):
  # Get sigmaY from the Gaussian fit to distance*sin(wd)
  #
  # Args: 
  # STR : time series
  # distance : distance from measurement to source (m)
  #
  # Returns: 
  # sigmay from Gaussian fit
  STR["theta"] = CenterWindOnPlume(STR)
  STRnew = STR[(STR["theta"] >= -90) & (STR["theta"] <= 90)] #subsetting 
  # Calculate Cross-wind Distance from Center of the Plume
  sinTheta = np.sin(STRnew["theta"]*pi/180)
  STRnew["Ly"] = distance
  LyMax = STRnew["Ly"].max(skipna = True)
  LyMin_10 = STRnew["Ly"].min(skipna = True)/10
  Ly_Min_Max = LyMax - LyMin_10
  binWidth = round(Ly_Min_Max,1)
  # Bin concentrations by Ly
  LyMin = STRnew["Ly"].min(skipna = True)
  minBin = floor(LyMin)
  maxBin = ceil(LyMin)
  ch4_thetabin = binConc(STRnew, binWidth = binWidth, binCutFilter=1, colName = "Ly") 
#  # Fit Gaussian Function
  params = ParamsFromGauss(ch4_thetabin, ch4_thetabin, 
                           ch4_thetabin, Skip , Test , STRfile)
 # input variables ->    # (ch4_thetabin["center"], ch4_thetabin["ch4"], 
                         # ch4_thetabin["n"],Skip,Test,STRfile)
#  CH4maxfit = params["a1"]
#  sigyfit = np.sqrt(params["sigma2"])
#  mu = params["mu"]
#  sigyerror = params["sigerror"]
#  sigyTtest = params["sigTtest"]
#  sigyPtest = params["sigP"]
#  # Plot Gaussian Fit
#  ch4_thetabin_min = np.min(ch4_thetabin["center"])
#  ch4_thetabin_max = np.max(ch4.thetabin["center"])
#
#  x = np.arange(ch4_thetabin_min, ch4_thetabin_max, 1)
#  gauss = CH4maxfit * np.exp(-0.5 * ((x - mu)**2)/(sigyfit**2))   
#  gauss.df <- data.frame(center=x, ch4=gauss)
#  
#  g1 <- ggplot(ch4.thetabin, aes(x=center, y=ch4), col="black") +     
#    geom_point(data = STRnew, aes(y=CH4, x=Ly), alpha=0.3, col="blue")+
#    geom_point(size=3) +
#    geom_line(data = gauss.df, col="red") +
#    theme_bw(base_size=16) +
#    xlab("WD") +
#    ylab("CH4 above background") + labs(title=as.character(Test)) 
#
#  # also plot the time series
#  g2 <- ggplot(data = STR, aes(x=center, y=ch4), col="black") +     
#    geom_line(data = STR, aes(y=CH4, x=Timeseries), alpha=0.3, col="black")+
#    theme_bw(base_size=16) +
#    xlab("Time") +
#    ylab("CH4 above background") +labs(title=as.character(Test)) 
#  
##  print(g1) 
##  print(g2)
#
##  multiplot(g1, g2, cols=2)  # cols means two columns
#  ggsave(filename = 
#           paste(basename(file_path_sans_ext(STRfile)), ".jpg", sep=""), 
#         plot = g1)
#  ggsave(filename=paste(basename(file_path_sans_ext(STRfile)),"time",".jpg",sep=""),
#        plot = g2)
  return();  #sqrt(params$sigma2)) # return this list(sigyfit=sigyfit, CH4maxfit=CH4maxfit, sigyerror=sigyerror,sigyTtest=sigyTtest,sigyPtest=sigyPtest)
def func(a,xvar,mu,sigma):
  x = a*np.exp(-1/2*(xvar-mu)**2/(sigma**2))
  return(x);
def ParamsFromGauss(xvar, yvar, BinWeights, Skip, Test, STRfile, qaCheck = False):
  # Calculate parameters for Gaussian fit using weighted, 
  # non-linear least squares
  #
  # Args: 
  # xvar : independent variable for fit
  # yvar : dependent variable for fit 
  # BinWeights : weights for fit
  #
  # Returns:
  # a1: fitted peak concentration
  # mu : fitted center of plume
  # sigma2 : fitted sigma2 (shape parameter)
  # use skip flag to skip the best-fit for non-converging files
  # pick one of the two following lines:
  
  #skip<-0 # run the best fit
#  skip<-10 #don't run the best fit
  
  # If no BinWeights given weight all points equally
  if (BinWeights[1]) != None: 
      BinWeights = np.repeat(1, len(xvar))
# Set initial values for gaussian fit estimation
  pylab.plot(xvar,yvar)
  ymax = np.max(yvar)
  mu0 = np.where(yvar == ymax,yvar,xvar)
  a0 = np.max(yvar)
  # Fit Gaussian curve to wd bins
  # for non-converging files skip the best-fit
  sigma = 10
  if all(Skip < 5):
#    plot(yvar,yvar) 
      y = a0*np.exp(-1/2*(xvar-mu0)**2/(sigma**2))
      fit = np.polyfit(xvar, y, 3)
 
#  Outstats =  fit["coef"]
#  print (Outstats)
#  mu = fit["coeff[1, 1]"]
#  sigma2 = (fit["coeff[2, 1]"])**2 
#  a1 = fit["coeff[3, 1]"]
#  sigerror = fit["coeff[2,2]"]
#  sigTtest = fit["coeff[2,3]"]
#  sigP = fit["coeff[2,4]"]
  
  # try getting best fit info for nls
  print(fit) 
  y = a0*np.exp(-1/2*(xvar-mu0)**2/(sigma**2))
  fit = np.polyfit(xvar, y, 3)
  
  
#  ok.res <- nlsResiduals(test_nls)
#  #plot(ok.res)
#  test.nlsResiduals(ok.res)
##  T95<-ok.res$std95
##  print (T95)
##  write.table(coef(nlsResiduals(test_nls)), file="statsout.csv") #, sep="n",append=TRUE)
##  print(coef(ok.res))
# 
#    RSS <- sum(resid(test_nls)^2)
#  
#  sh <- shapiro.test(resid(test_nls)) 
#  shW <- sh$statistic # W term
#  shP <- sh$p.value   # p term 
##  skip<-0
#  } 
  if all(Skip>5):
    # for non-converging files, use these instead
      mu = mu0
      a1 = a0
      sigma2 = 10.
      sigerror = -99.
      sigTtest = -99.
      sigP = -99.
      RSS = -99.
      sh = -99.
      shW = -99.
      shP = -99.
#  
#  # QA check - if either tail does is not lower than half of the peak
#  # concentration, throw an error 
#  if (qaCheck):
#    ch4.bin <- data.frame(xvar=xvar, yvar=yvar)
#    ch4.bin <- ch4.bin[order(ch4.bin$xvar), ]
#    gauss <- a1 * exp(-0.5 * ((ch4.bin$xvar - mu)^2/sigma2)) 
#    if (gauss[1] > max(ch4.bin$ch4)/2 | 
#          gauss[length(gauss)] > max(ch4.bin$ch4)/2) {
#      stop("Does not pass QA check")
#    }
#  }
  return();#return list(a1 = a1, mu = mu, sigma2 = sigma2, sigerror=sigerror, sigTtest=sigTtest, sigP = sigP, RSS = RSS, sh = sh, shW = shW, shP = shP

# multiplot Function
#multiplot <- function(..., plotlist = NULL, file, cols = 1, layout = NULL) {
#  require(grid)
#  
#  plots <- c(list(...), plotlist)
#  
#  numPlots = length(plots)
#  
#  if (is.null(layout)) {
#    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
#                     ncol = cols, nrow = ceiling(numPlots/cols))
#  }
#  
#  if (numPlots == 1) {
#    print(plots[[1]])
#    
#  } else {
#    grid.newpage()
#    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
#    
#    for (i in 1:numPlots) {
#      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
#      
#      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
#                                      layout.pos.col = matchidx$col))
#    }
#  }
#}

#open the CSV file with data
#with open('NYCFileNames.csv','r') as csvfile:
#    NYCreader = csv.reader (csvfile, delimiter=' ', quotechar = '|')
#    for row in NYCreader:
#        print(row)
FileName = []
#f = open( 'NYCFileNames.csv', 'r' ) #open the file in read universal mode
f = pd.read_csv('NYCFileNames.csv', header = 0)
sonicHeight = 2.51

# Sampling frequency changes
samplingFreq = 10 # Hz
lagTime = 3.8 # seconds Picarro in NYC, NOv, 2016
k = 0.4 # von Karman constant
g = 9.8 #gravitational acceleration m/s^2
rho = 1.292 # kg/m3 at 0 deg C and 1 atm
Cp = 1005 # specific heat of air J/kg/K
# Initialize progress bar

i = 0

# Create data.frame to store results
results = pd.DataFrame({"file":[FileName],"Test":[None], "SensorHt":[None], "SourceHt":None,
                   "distancem":None,"Lat":None, "Long":None,"wsbar":None, "Tbar":None,
                   "TbarK":None, "pbar":None, "ubar":None,"vbar":None,  "wbar":None, 
                   "sigu":None, "sigv":None, "sigw":None,"turbhoriz":None, "turbvert":None,
                   "sigT":None, "Hflux":None, "ustar":None, "Lmonin":None, "CH4back":None,"CH4max":None,
                   "CH4maxfit":None, "CH4maxbin":None, "CH4max98":None, 
                   "stabilityClass":None, "sigyPG":None, "sigzPG":None, "sigyfit":None,
                   "sigyturb":None, "sigzturb":None, "reflectionPG":None, "reflectionturb":None,
                   "CH4RatePG":None, "CH4RatePGsigzturb":None,"CH4RateTurbCmaxbin":None,
                   "CH4RatePGsigyfit":None, "sigyerror":None, "sigyTtest":None,"sigyPtest":None})

fileName = f[[4]] # FileName
sensorHt = f[[5]] # SensorHt
sourceHt = f[[6]] # SourceHt
distanceFeet = f[[7]]*.3048 # Distance Feet
test = f[[8]] # Test
skip = f[[9]] # Skip

for i in range(0,42):
   fileName_list = fileName.loc[[i]]
   fileName_str = fileName_list.to_string(header = False, index = False)
   STRraw = pd.read_csv(fileName_str,error_bad_lines=False,skiprows=6)
   i+=1
   print(5)
  
STRraw.columns = ['TimeStamp', 'GPSLatitude', 'GPS.Longitude', 'satellites', 'GPSTime',
                     'CompassHeading', 'u', 'v', 'w', 'SonicTemp',
                     'CO2.ppm', 'CH4ppm',  'press', 'CH4.gatos', 'Skip'] 
 # Pre-processing
# cant find times equiv in python
  # mean position
Lat = STRraw[[1]].mean()
Long = STRraw[[2]].mean()
u = STRraw[[6]]
v = STRraw[[7]]

STRraw["wd_sonic"] = np.arctan2(-u,-v)*180/pi

mean_wd_raw = np.arctan2(u.mean(),v.mean())*180/pi

#STRraw$ws_sonic <- sqrt(STRraw$u^2+STRraw$v^2)
u_squ = u*u
v_squ = v*v
u_v = np.add(u_squ, v_squ)
STRraw["ws_sonic"] = np.sqrt(u_v)

#STR Align
STR_align = timeAlign(STRraw, lagTime*samplingFreq, 11)
# use sonic for wind speed and direction
STR_align["wd_met"] = STRraw["wd_sonic"]
STR_align["ws_met"] = STRraw["ws_sonic"]

# Rotate Sonic to Streamlined Coordinates
STR = rotateSonic(STR_align,pi)
# set background at the 5th percentile, but use 25% point to determine which data
# to use for binning.  
  
CH4_5 = np.percentile(STR["CH4ppm"], 5)
CH4_5sort = [k for k in STR["CH4ppm"] if k <= CH4_5 ]
CH4_5mean = np.mean(CH4_5sort)
STR["CH4back"] = CH4_5mean
   
CH4_25 = np.percentile(STR["CH4ppm"], 25)
CH4_25sort = [k for k in STR["CH4ppm"] if k <= CH4_25 ]
CH4_25mean = np.mean(CH4_25sort)
STR["filter"] = CH4_25mean
   
STR["CH4"] = np.where(STR["CH4ppm"] > CH4_25mean, (STR["CH4ppm"] - STR["CH4back"]), None)

#  mean_wd_rot <- atan2(mean(STR$u),mean(STR$v))*180./pi
u_mean = np.mean(STR["u"])
v_mean = np.mean(STR["v"])

mean_wd_rot = np.arctan2(u_mean,v_mean)*180/pi
STR["wd_sonic"] = CenterWindOnPlume(STR)
mean_wd_center = np.mean(STR["wd_sonic"])

# plot time series of CH4
STR["Timeseries"] = STR_align["TimeStamp"]

pbar = np.mean(STR["press"]) 
Tbar = np.mean(STR["SonicTemp"]) # in deg C
wsbar = STR["ws_sonic"].mean(skipna = True)
turbhoriz = STR["ws_sonic"].std(skipna = True)/wsbar#horizontal turbulent intensity
turbvert = STR["w"].std(skipna = True)/wsbar #vertical turbulent intensity
CH4max = np.max(STR["CH4"]) # max CH4 from raw data with background subtracted (instantaneous)
  
# Pasquill Stability Class 
stabilityClass = GetStability(STR)
# get means and standard deviations for winds and temperature
wbar = STR["w"].mean(skipna = True)#should be close to zero after rotation
ubar = STR["u"].mean(skipna = True)#should be close to zero after rotation
vbar = STR["v"].mean(skipna = True)#should be equal to wsbar after rotation
sigu = STR["u"].std(skipna = True)
sigv = STR["v"].std(skipna = True)
sigw = STR["w"].std(skipna = True)     

u_w = STR["u"]*STR["w"]
v_w = (STR["v"]-vbar)*STR["w"]
sonic_Tbar = (STR["SonicTemp"]-Tbar)*STR["w"]

uwbar = u_w.mean(skipna = True) #uw cross product
vwbar = v_w.mean(skipna = True) #vw cross product
sigT = STR["SonicTemp"].std(skipna = True)
Hflux = sonic_Tbar.mean(skipna = True) #kinematic heat flux (m*K/s)
ustar = (uwbar*uwbar + vwbar*vwbar)**0.25 # friction velocity m/s
TbarK = Tbar+273.15 #T in Kelvin
Lmonin = (-TbarK*(ustar**3)/(k*g*Hflux))  # Monin-Obuhkov length (m)
HfluxWatts = Hflux*rho*Cp # heat flux in W/m2 

############################################################################# 
  
  # Bin concentration by wind direction
  #  11-01-2016   change bin filter from 0.02 to 0.002 to get convergence
nrow = len(STR)
ch4wdbin = binConc(STR, 10, binCutFilter=0.002*nrow)

# Only include downwind bins
#center = 0
#ch4wdbin = ch4wdbin[center >= -90 & center <= 90]

# find max binned concentration
CH4maxbin = np.max(ch4wdbin) #max with background subtracted (Should be "ch4")

# for now set 98% max to max binned CH4
CH4max98 = CH4maxbin  

pgsigma = pd.read_csv('pgsigma.csv', header = 0)

  # Using Pasquill-Gifford (PG) table for sigmay, sigmaz
psgConst = GetPGsigma(pgsigma, distanceFeet, stabilityClass)
sigyPG = pgsigma["sigmay"] #originally pgsigmay
sigzPG = pgsigma["sigmaz"]  #originally pgsigmaz   

  # do Gaussian best-fit to get sigma-y and Cmax-fit
sigyout = GetSigmaYdata(STR, distanceFeet,test,skip,fileName)
#sigyfit = sigyout["sigyfit"]
#CH4maxfit = sigyout["CH4maxfit"]
#sigyerror = sigyout["sigyerror"]
#sigyTtest = sigyout["sigyTtest"]
#sigyPtest = sigyout["sigyPtest"]   #fix the function  

  # calculate sigma's using turbulence data
sigyturb = turbhoriz*distanceFeet
sigzturb = turbvert*distanceFeet
H = sourceHt  #near the ground
z = sensorHt          

#Use sigma's and Cmax's to calculate CH4 emission rates
#method33a(1)
reflectionPG = np.exp(-0.5*((z+H)/sigzPG)**2) + np.exp(-0.5*((z-H)/sigzPG)**2)
#Cmaxfitgrams = CH4maxfit*1e-6*16./(0.022414*1013./pbar*TbarK/273.15)
#CH4RatePG = (Cmaxfitgrams*2.*pi*wsbar*sigyPG*sigzPG)/reflectionPG 

#method33a-revised(2)--use sigmas from turbulence with Cmax-bin
reflectionturb = np.exp(-0.5*((z+H)/sigzturb)**2) + np.exp(-0.5*((z-H)/sigzturb)**2)
Cmaxbingrams = CH4maxbin*1e-6*16/(.022414*1013/pbar*TbarK/273.15)

CH4RateTurbCmaxbin = (Cmaxbingrams*2*pi*wsbar*sigyturb*sigzturb)/reflectionturb
                     
##method33a-revised(3)--use sigma-y fit and sigma-z PG with Cmaxfit
#CH4RatePGsigyfit = (Cmaxfitgrams*2*pi*wsbar*sigyfit*sigzPG)/reflectionPG
#                   
##method33a using turbulent sigmaz with PG sigma Y and best-fit Cmax
#CH4RatePGsigzturb = (Cmaxfitgrams*2*pi*wsbar*sigyPG*sigzturb)/reflectionturb  
  
# Store output
results["SensorHt"] = sensorHt
results["SourceHt"] = sourceHt
results["Test"] = test 
results["distancem"] = distanceFeet
results["Lat"] = Lat
results["Long"] = Long
results["Tbar"] = Tbar
results["TbarK"] = TbarK
results["pbar"] = pbar
results["wsbar"] = wsbar
results["turbhoriz"] = turbhoriz
results["turbvert"] = turbvert
results["wbar"] = wbar
results["ubar"] = ubar
results["vbar"] = vbar
results["sigu"] = sigu
results["sigv"] = sigv
results["sigw"] = sigw
results["sigT"] = sigT
results["ustar"] = ustar
results["Hflux"] = HfluxWatts
results["Lmonin"] = Lmonin
#results["CH4back"] = CH4back
results["CH4max"] = CH4max
#results["CH4maxfit"] = CH4maxfit
results["CH4maxbin"] = CH4maxbin
results["CH4max98"] = CH4max98
results["stabilityClass"] = stabilityClass
results["sigzPG"] =  sigzPG
results["sigyPG"] = sigyPG
#results["sigyfit"] = sigyfit
results["sigyturb"] = sigyturb
results["sigzturb"] = sigzturb
results["reflectionPG"] = reflectionPG
results["reflectionturb"] = reflectionturb
       
  #convert to scfh at 60 F, 1 atm
#results["CH4RatePG"] = CH4RatePG/(19.2)*3600.
#results["CH4RatePGsigzturb"] = CH4RatePGsigzturb/(19.2)*3600.
results["CH4RateTurbCmaxbin"] = CH4RateTurbCmaxbin/(19.2)*3600.
#results["CH4RatePGsigyfit"] = CH4RatePGsigyfit/(19.2)*3600.
#results["sigyerror"] = sigyerror
#results["sigyTtest"] = sigyTtest
#results["sigyPtest"] = sigyPtest
print(sensorHt)
results.to_csv('outfile.csv')  
                 
import rpy2.robjects as psigmas
psigmas.r['load']("pgsigma.RData")

