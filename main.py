import alsaaudio, time, audioop
import numpy as np
import scipy, scipy.fftpack, scipy.cluster.vq, scipy.signal
import matplotlib.pyplot as plt
import math, sys, random, pickle, glob
import subprocess
import random

DEBUG = True

SAMPLE_RATE = 16000
SIZEOF_PERIOD = 176400
SAMPLE_MODE = alsaaudio.PCM_NORMAL 
FRAME_SIZE = 320
FRAME_SPACE = 100



def sgn(v):
	if v >= 0.0:
		return 1.0
	else:
		return -1.0


def preemphasis(shat):
	"""This highpass filters the signal shat"""
	print "Preemphasis",len(shat)
	s1 = scipy.signal.lfilter([1,-0.95],1.0,shat)
	return s1

def _VAD_trigger(s1,L):
	"""This is only used inside the VAD method"""
	W = []
	Sc = 1000.0
	Linv = 1.0/float(L)

	for m in range(10*L,L*20+1,L):
		idx = range(m-1,m+L-1)
	 	E = np.sum( np.square( s1[idx] ) )
	 	P = E*Linv
	 	Z = 0.0
	 	for n in idx: Z = abs( sgn(s1[n]) - sgn(s1[n-1]) )
	 	Z = Z*Linv*0.5
	 	W.append( P*(1.0 - Z)*Sc )


	my = np.mean(W)
	sig = np.std(W)
	
	tw = my + (0.2*(sig**(-0.8)))*sig

	if tw < min(s1) or tw > max(s1):
		raise Exception("VAD threshold is out of bounds")

	return tw


def VAD(s1,L):
	""" Void Activation Detection accoring to [2] 
	L is the blocklength (FRAME_SIZE) in number of samples."""
	print "VAD"
	Sc = 1000.0
	P = []
	Z = []
	x1 = []
	dbg_removed_blocks = 0

	# Compute the trigger
	tw = _VAD_trigger(s1,L)

	print "VAD trigger:",tw

	if len(s1) < L*20+1:
		print "WARNING: Sample is too short to make VAD trigger"

	# Comput VAD
	for m in xrange(L*10,len(s1),L):
	
		idx = range(m-L+1,m)

		Z = 0.0

		for n in idx:
			Z += abs( sgn(s1[n]) - sgn(s1[n-1]) )/2.0

		E = sum(s1[idx]*s1[idx])
		P = E/L
		Z = Z/L

		W = P*(1.0 - Z)*Sc

		
		if W > tw:
			x1 += list(s1[idx])
		else:
			dbg_removed_blocks += 1

	print "-> Removed samples",dbg_removed_blocks*L
	return x1



def frame_blocking_windowing(processed_speech):
	""" Preprocessing stage """
	print "Frame blocking and windowing"		
	# FRAME BLOCKING and WINDOWING
	P = 100
	K = 320

	wk = 0.54-0.46*np.cos( (2*np.pi*np.array(range(0,K)))/(K-1) )
	x1 = np.array(processed_speech)	
	
	xk = []
	for i in range(0,len(x1),K):
		if len(x1)-i < K:
			break
		else:
			xk.append(x1[i:i+K]*wk)

	print "-> New shape: ",len(xk),K

	return xk
	

def mel_filter(X,df,ll,ul,NFFT):
	"""Compute X filtered with a triangle starting a ll up to ul """

	start_at = int(ll/df)
	peak_at = int(((ul-ll)/2.0)/df) + start_at
	end_at = int(ul/df)

	
	mk = 0.0
	h = 1.0/(float(peak_at-start_at+1))
	for n,i in zip(range(start_at,peak_at+1),xrange(0,peak_at-start_at+1)):
		if h*i>1.0 or h*i < 0.0:
			raise Exception("h*i=",h*i)
		mk += (h*i)*X[n]

	h = 1.0/(float(end_at-peak_at+1))
	for n,i in zip(range(peak_at,end_at+1),xrange(0,end_at-peak_at+1)):
		t = (1.0-h*i)
		if t>1.0 or t < 0.0:
			raise Exception(t,peak_at,end_at,i,h)
		mk += (1.0-h*i)*X[n]
	
	return mk


def mel_filter_bank(freq,K_coeff,NFFT):

	N  = NFFT
	Hn = np.zeros((N,N))
	
	df = freq[1]-freq[0]

	r = 0
	for ll,ul in K_coeff:
		start_at = int(ll/df)
		peak_at = int(((ul-ll)/2.0)/df) + start_at
		end_at = int(ul/df)

		h = 1.0/(float(peak_at-start_at+1))


		for n,i in zip(range(start_at,peak_at+1),xrange(0,peak_at-start_at+1)):
			Hn[r,n] = h*i
		for n,i in zip(range(peak_at,end_at+1),xrange(0,end_at-peak_at+1)):
			Hn[r,n] = 1.0-h*i
		r+=1

	return Hn


def mel_cepstrum(x2,K):
	""" Transformation of the signal x2 to Mel-Cepstrum accoring to [2] """
	print "Mel-Cepstrum"

	# ZERO PADDING
	NFFT = int(pow(2, np.ceil(math.log(K, 2))))

	print "-> Data was padded: ",K," -> ",NFFT

	# Mel-Scaled Filterbank, full of magic numbers
	K20_filter = [ 
	(0.0,154.759),
	(77.3795,249.2458),
	(163.3126, 354.1774),
	(258.745,470.7084),
	(364.7267,600.121),
	(482.4239,743.8391),
	(613.1315,903.4442),
	(758.2878,1080.6923),
	(919.4901,1277.5338),
	(1098.5119,1496.1345),
	(1297.3232,1738.8999),
	(1518.1115,2008.501),
	(1763.3063,2307.9044),
	(2035.6053,2640.4045),
	(2338.0049,3009.6599),
	(2673.8324,3419.7335),
	(3046.7829,3875.1375),
	(3460.9602,4380.8829),
	(3920.9215,4942.5344),
	(4431.728, 5566.272)
	]

	# ----------------------------------
	# The final Mel Filter Cepstral Coefficients
	# have len(K20_filter) coefficients (rows) and 
	# the operation is performed on each window.
	# ----------------------------------
	

	NUM_WINDOWS = len(x2)
	print "NUM WINDOWS",NUM_WINDOWS

	Q = 14	
	MFCC = np.zeros((Q,NUM_WINDOWS))

	plt.subplot(222)
	plt.title("Mel cepstrum coefficients")

	win_id = 0
	for win in x2:

		# DFT	
		X2 = np.absolute(scipy.fftpack.fft( win, NFFT ))	
		freq = scipy.fftpack.fftfreq(NFFT, 1.0/SAMPLE_RATE)
	
		X2 = X2[len(X2)/2:]
		freq = freq[ freq.shape[-1]/2: ]

		df = freq[1]-freq[0]
		K = len(K20_filter)	
		
		mks = np.zeros(K)
		for i in xrange(0,K):
			ll,ul = K20_filter[i]
			mks[i] = mel_filter(X2, df, ll, ul, NFFT)

		plt.plot(mks)

		c = np.zeros(Q)
		invc = np.zeros((Q,K))
		for q in xrange(0,Q):
			for k in range(0,K):
				c[q] += np.log(mks[k])*np.cos( (np.pi*q*(2*k+1)) / (2*K) )
				invc[q,k] = np.cos( np.cos( (np.pi*q*(2*k+1)) / (2*K) ) )
		
		# IDCT
		MFCC[:,win_id] = c
		win_id += 1

	return MFCC
	

def postprocess(x3):
	""" Just normlize to get zero mean. """
	print x3.shape
	fmy = np.mean(x3)
	print fmy
	col = 0
	for col in xrange(0,x3.shape[-1]):
		x3[:,col] -= fmy
	return x3

def l2norm(v1,v2):
	return np.linalg.norm( v1-v2  )

class DynamicTimeWarper:

	def __init__(self,distance_metric=l2norm):

		self.observations = []
		self.distfun = distance_metric

	def read_from_path(self,path):

		files = glob.glob(path+"/*.fv")

		if path == None:
			print "DynamicTimeWarper: read_from_file: You forgot to spec. the path"
			return
		
		for fname in files:
			try:
				fobj = open(fname,"r")
				samples = np.asarray(pickle.load(fobj))
				fobj.close()
				self.add(fname,samples)
			except IOError, e:
				print "DynamicTimeWarper: read_from_file: Unable to open file ",fname



	def add(self, name, fv): 
		name = name[-name[::-1].find("/"):]
		self.observations.append((name,fv))

	def DTWdistance(self, w, x):

		Tw1 = w.shape[-1]
		Tx1 = x.shape[-1]

		Xi = []
		for i in range(0,Tw1):
			Xi.append([])
			for j in range(0, Tx1):
				Xi[-1].append((-1,-1))

		dj = np.zeros( Tw1 )
		djm1 = np.zeros( Tw1 )

		Xi[0][0] = (-1,-1)
		dj[0] = self.distfun(w[:,0],x[:,0])

		for i in range(1,Tw1):
			dj[i] = self.distfun(w[:,i],x[:,0]) + dj[i-1]
			Xi[i][0] = (i-1,0)

		# iteration
		for j in range(1,Tx1):

			# Swap arrays
			tmp = djm1
			djm1 = dj
			dj = tmp

			# first point
			dj[0] = self.distfun( w[:,0],x[:,j] ) + djm1[0]
			Xi[0][j] = (0,j-1)

			for i in range(1,Tw1):

				# Optimization
				dj[i] = min( [ djm1[i]+self.distfun(w[:,i], x[:,j]), djm1[i-1]+2.0*self.distfun(w[:,i], x[:,j]), dj[i-1]+self.distfun(w[:,i], x[:,j]) ] )

				# Tracking path decisions
				paths = [(i,j-1),(i-1,j-1),(i-1,j)]
				k,l = paths.pop(0)
				mini = min( [ djm1[k]+self.distfun(w[:,k], x[:,l]), djm1[k-1]+2.0*self.distfun(w[:,k], x[:,l]), dj[k-1]+self.distfun(w[:,k], x[:,l]) ] )
				for k,l in paths:
					val = min( [ djm1[k]+self.distfun(w[:,k], x[:,l]), djm1[k-1]+2.0*self.distfun(w[:,k], x[:,l]), dj[k-1]+self.distfun(w[:,k], x[:,l]) ] )
					if val < mini:
						mini = val
						Xi[i][j] = (k,l)

		# Termination

		DWT = dj[Tw1-1]

		# Backtracking

		# init 
		i = Tw1-1
		j = Tx1-1

		path = [Xi[i][j]]

		while Xi[i][j] != (-1,-1):
			i,j = Xi[i][j]
			path.append( (i,j))

		#print path

		score = 0.0
		for (i,j) in path:
			score += self.distfun( w[:,i], x[:,j] )

		return score

	def recognize(self, obs_in):

		results = []
		for (name,obs) in self.observations:
			#print "Checking", type(obs)
			results.append( (name, self.DTWdistance(obs_in, obs)) )

		return sorted(results, key=lambda k: k[1])


class SoundRecon:
	""" This is a whole word speech recognizer. A whole word recognitioner 
	constructs HMMs from simple HMMs by concatenation. Paramters in the 
	HMM are estimated using the Baum-Welch. The most probable path 
	through the HMM is computed using the Viterbi (forced) algorithm. 


	Steps in recognizing speech:

	Audio Processor (input audio, output feature vector)
	===============
	Speech is processed per window from the audi sampler. The window 
	is sent through a Linear Predictive Codeing (LPC) which outputs 
	n cepstral coefficients. Below are the steps in the LPC:

	1. Preprocessing, 
	2. Noise Cancelling by (Spectral substraction)
	3. Preemphasis H(z) = 1 - 0.95z^-1 <=> s_1(n) = sum_k h(k)shat(n-k)
	4. Voice Activation Detection (VAD) ...
	5. Frame blockig and Windowing using: w(k) = 0.54-0.46*(2*pi*k)/(K-1)
	----------------------------------------------------------------------
	=	Speech vector

	Feature extraction
	==================
	  Mel-Cepstrum
	+ Post Processing
	------------------
	= Feature vector


	Model training
	==============
	Baum-Welch

	"""

	def __init__(self):
		
		# Open the device in nonblocking or blocking capture mode
		# alsaaudio.PCM_NORMAL
		# alsaaudio.PCM_NONBLOCK
		names = alsaaudio.cards()
		self.source = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, SAMPLE_MODE,card=names[-1]) 

		# Mono, 44100 Hz, 32 bit little endian samples
		self.source.setchannels(1) # no you cant use stereo
		self.source.setrate( SAMPLE_RATE )
		self.source.setformat(alsaaudio.PCM_FORMAT_S16_LE)
		self.source.setperiodsize( SIZEOF_PERIOD )

		self.Or = None

	def audio_processor(self, speech, filename="sample.fv"):
		""" Preprocessing stage """
		#
		# TODO: first filter here, x -> shat
		#
		plt.subplot(221)
		plt.plot(speech)
		plt.grid()
		plt.title("Speech signal")

		print "Speech length",len(speech)
		s1 = preemphasis(speech)
		x1 = VAD(s1, FRAME_SIZE)

		plt.subplot(223)
		plt.plot(x1)
		plt.grid()
		plt.title("Voice activation filtered speech")
		
		#	Return
		x2 = frame_blocking_windowing(x1)
		x3 = mel_cepstrum( x2, FRAME_SIZE )
		O = postprocess(x3)

		plt.subplot(224)
		plt.plot(O.T)
		plt.grid()
		plt.title("Mel coefficients")
		plt.show()

		print "Dumping feature vector to file ",filename
		fp = open(filename,"w")
		pickle.dump(O,fp)
		fp.close()
		
		return O


	def capture_word(self):
		
		print "Recording 2 sec"	
		self.sample = np.array("h",[])
		buff = []
		nf = 0
		
		rec = 0
		while rec < 2.0:
			length,data = self.source.read()
			if length>0:
				# Data is unpacked from a string representation. The FFT is 
				# performed on 10 ms intervals 
				buff.append(data)
				nf += length
				rec += FRAME_SIZE*1.0/SAMPLE_RATE
			else:
				if SAMPLE_MODE == alsaaudio.PCM_NORMAL:
					print "No data, WTF?"

		print "Recorded " + str( rec )

		#
		# Recording complete.
		# Start spectral transformation by Linear Predictive Coding (LPC)
		#

		sig = list(4.0/float(2<<16)*np.fromstring("".join(buff),np.int16,nf))

		return sig

	def load_feature_vectors(self, path="./storage/"):
		
		fv_files = glob.glob(path+"*.fv")

		if len(fv_files) < 1:
			print "Found no saved feature vectors in",path
			exit(1)

		print "-> Loading sample files ",len(fv_files)
		
		R = 0
		samples = []

		for fv in fv_files:
			print path+fv
			try:
				fp = open(fv,"r")
				samples.append(np.asmatrix(pickle.load(fp)))
				fp.close()
				R += 1
			except IOError, e:
				print "Unable to open file ",fv

		T,D = samples[0].shape

		self.Or = np.zeros((R,D,T))

		MOS = MultiObsSeq()

		for r in xrange(0,R):
			MOS.add(samples[r].T)
			self.Or[r,:,:] = samples[r].T


		R,D,T = self.Or.shape
		print "|- Samples: ",R
		print "|- Feature len: ",D
		print "|- Num. of states: ",T

			

	def train_model(self):

		# --------------------------
		# Setup initial guess
		# --------------------------
		R,D,T = self.Or.shape

		self.B = Emission_matrix(3,self.Or)
		self.pi = np.zeros((T,1))
		self.pi[0] = 1.0
		self.A = np.zeros((T,T))
		for j in xrange(2,T):
			self.A[j-2,j-2] = 1.0/3.0
			self.A[j-2,j-1] = 1.0/3.0
			self.A[j-2,j  ] = 1.0/3.0

		self.A[-2,-2] = 0.5; self.A[-2,-1] = 0.5;
		self.A[-1,-1] = 1.0


		[A,B,pi] = baum_welch(self.A,self.B,self.pi,self.Or)
		return [A,B,pi]

		
		

	def log_to_file(self,xydata,fname="sample.dat"):
		fp = open(fname,"w")
		fp.write( "\n".join([str(p[0])+" "+str(p[1]) for p in xydata]) )
		fp.close()

	def __save_recording(self,word,FFT):		
		fp = open("resources/"+word+".samp","w")
		fp.write(FFT.tostring())
		fp.close()



def plot_word_feature_vector(path="./storage/"):
	samples = []
	R = 0
	fv_files = glob.glob(path+"*.fv")
	legends = []
	for fv in fv_files:
		print fv
		try:
			fp = open(fv,"r")
			data = list(pickle.load(fp))
			samples.append(data)
			legends.append(fv)
			fp.close()
			R += 1
		except IOError, e:
			print "Unable to open file ",fv

		plt.plot(data)

	print legends

	plt.title("Feature vectors of "+path+"*.fv")
	plt.legend(legends)
	plt.grid()
	plt.show()


if __name__ == "__main__":

	SR = SoundRecon()
  	speech = SR.capture_word()
	obs = SR.audio_processor(speech)
	DTW = DynamicTimeWarper()
	DTW.read_from_path("./storage")

	# testing
	for name,obs in DTW.observations:
		ranking = DTW.recognize( obs )

		n,w = ranking[0]

		if "foo" in n:
			subprocess.call(["mpg123","effects/button-3.mp3"])
		print name,"==",n

