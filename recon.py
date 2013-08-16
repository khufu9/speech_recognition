#!/usr/bin/python
""" This work was perfomed using these two references:
	[1] Statistical Methods for Speech Recognition, Jelinek
	[2] Speech Recognition using Hidden Markov Model - 
	performance evaluation in noisy environment by (M. Nilsson M. Ejnarsson) 2002.

	Dimensionality break-down:

	* Feature extraction

	
"""

import alsaaudio, time, audioop
import numpy as np
import scipy, scipy.fftpack, scipy.cluster.vq, scipy.signal
import matplotlib.pyplot as plt
import math, sys, random, pickle, glob
import random

DEBUG = True

# A frame is a sample per channel
# Period size is in HERTZ!

SAMPLE_RATE = 16000
#SAMPLE_RATE = 8000
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
	# s1 = np.zeros((len(shat)))
	# s1[0] = shat[0]
	# for n in range(1,len(shat)):
	# 	s1[n] = shat[n]-0.95*shat[n-1]
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
	"""Compute X filtered with a triangle starting a ll up to ul"""
	"""
	df = SAMPLE_RATE*1.0/(NFFT/2+1)
	h = 1.0/((ul-ll)/2.0)
	w = 0.0
	start = int(ll/df)
	stop = int(ul/df)
	mid = (stop-start)/2

	mk = 0.0
	for i in range(start,mid):
		mk += X[i]*((i-start)*h)
	for i in range(mid,stop):
		mk += X[i]*(1.0-(i-mid)*h)

	return mk
	"""
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

	""" Plot the filter vector (recommeded to do it one filter at a time since
		the sum of filters may look like a dinosour's back)
	plt.figure()
	plt.plot(Hs)
	plt.show()
	"""

	return Hn


def mel_cepstrum(x2,K):
	""" Transformation of the signal x2 to Mel-Cepstrum accoring to [2] """
	print "Mel-Cepstrum"

	# ZERO PADDING
	NFFT = int(pow(2, np.ceil(math.log(K, 2))))

	print "-> Data was padded: ",K," -> ",NFFT
	"""
	x2row = len(x2)

	X2 = np.zeros((x2row,2**b))
	ind = range(padding/2,padding/2+K)
	
	for i in range(0,x2row):
		X2[i][ind] = x2[i]
	"""

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


	# plt.subplot(224)
	# for co in invc:
	# 	plt.plot(co)
	# plt.title("Cosines")

	return MFCC
	

	

	# note N is now the number of wanted cepstrum coeffs. 
	# N = len(K20_filter) # i.e. 20 by 20

	# cs = np.zeros((N,1))
	
	# for n in xrange(0,N):
	# 	csm = np.sqrt(1.0/N)*lmks[0]*np.cos( 0.0 )
	# 	for k in xrange(1,N):
	# 	 	csm += np.sqrt(2.0/N)*lmks*np.cos( np.pi*(2.0*n+1.0)*k/(2.0*N) )

	# 	cs[n] = csm

	# Lifter
	# L = int(2.0/3.0*N)
	# for n in xrange(0,L):
	# 		cs[n] = cs[n]*(1.0+ (L-1)/2.0*np.sin( np.pi*n/(L-1)))
	# ch = cs

	# plt.subplot(313)
	# plt.plot(ch)

	# plt.show()

	#return MFCC


def postprocess(x3):
	""" Just normlize to get zero mean. """
	print x3.shape
	fmy = np.mean(x3)
	print fmy
	col = 0
	for col in xrange(0,x3.shape[-1]):
		x3[:,col] -= fmy
	return x3

class EmissionMatrix:
	c = None
	u =	None
	s = None
	O = None
	

	def __init__(self, obsMatrix):
		""" K-means clustering """

		self.O = obsMatrix
		N,M,R = self.O.shape
		
		# ----------------------------------
		# Make space for c,u and s
		# ----------------------------------
		self.c = np.zeros((M,3))	# a constant
		self.u = np.zeros((M,3,N))	# a vector
		self.s = np.zeros((M,3,N))	# a vector

		# ----------------------------------
		# For every emission of state j in N
		# ----------------------------------
		for j in xrange(0,M):
			# ----------------------------------
			# Get the observations from all 1st observations [:][1][:]
			# ----------------------------------
			obs = self.O[:,j,:]
			print obs.shape

			# ----------------------------------
			# Cluster centers then classify (vector quantitization)
			# kmeans2 expect M Observations of length N
			# ----------------------------------
			centroids, _ = scipy.cluster.vq.kmeans( obs.T, 3 )
			idx,_ = scipy.cluster.vq.vq(obs.T,centroids)

			# ----------------------------------
			# Compute c u and s
			# ----------------------------------
			idx = list(idx)
			self.c[j,0] = idx.count(0)/float(len(idx))
			self.c[j,1] = idx.count(1)/float(len(idx))
			self.c[j,2] = idx.count(2)/float(len(idx))
			
			# Codes for each cluster
			c1 = [i for i in range(0,len(idx)) if idx[i]==0]
			c2 = [i for i in range(0,len(idx)) if idx[i]==1]
			c3 = [i for i in range(0,len(idx)) if idx[i]==2]

			print "K-means cluster sizes:",len(c1),len(c2),len(c3)

			self.u[j,0,:] = np.mean(obs[:,c1],axis=1)
			self.u[j,1,:] = np.mean(obs[:,c2],axis=1)
			self.u[j,2,:] = np.mean(obs[:,c3],axis=1)

			
			cov0 = np.cov(obs[:,c1],rowvar=1)
			cov1 = np.cov(obs[:,c2],rowvar=1)
			cov2 = np.cov(obs[:,c3],rowvar=1)
			print "Cluster covariances",cov0,cov1,cov2
			# cov0 = np.diag(np.cov(obs[:,c1],rowvar=1))
			# cov1 = np.diag(np.cov(obs[:,c2],rowvar=1))
			# cov2 = np.diag(np.cov(obs[:,c3],rowvar=1))


			# ----------------------------------
			# Check for NaN then eps > diag[g]
			# ----------------------------------
			# if any(filter(lambda x:x!=x, cov0)): cov0 = [1e-5]*N
			# if any(filter(lambda x:x!=x, cov1)): cov1 = [1e-5]*N
			# if any(filter(lambda x:x!=x, cov2)): cov2 = [1e-5]*N

			# cov0 = [1e-5 if x < 1e-5 else x for x in cov0]
			# cov1 = [1e-5 if x < 1e-5 else x for x in cov1]
			# cov2 = [1e-5 if x < 1e-5 else x for x in cov2]


			self.s[j,0,:] = cov0
			self.s[j,1,:] = cov1
			self.s[j,2,:] = cov2



			if DEBUG:
				# plot all clusters and their mean in read
				plt.subplot(311)
				plt.plot(obs[:,c1][0])
				plt.subplot(312)
				plt.plot(obs[:,c2][0])
				plt.subplot(313)
				plt.plot(obs[:,c3][0])
				plt.show()

				plt.plot([cov0,cov1,cov2])
				plt.show()



	def get(self,j,obs):

		def project(diag):
			# ----------------------------------
			# diag is projected onto the diagonal of I
			# ----------------------------------
			D = np.zeros((diag.shape[0],diag.shape[0]))
			for d in range(0,diag.shape[0]):
				D[d,d] = diag[d]
			return D

		def mnd( self, x, u, s ):
			s = project(s)
			if x.shape != u.shape:
				raise RuntimeError("Multi-variate normal distribution: x and u has wrong dimensions:"+str(x.shape)+" "+str(u.shape))
			if s.shape[1] != s.shape[0]:
				raise RuntimeError("Multi-variate normal distribution: Sigma matrix is not square: "+str(s.shape))
			# ----------------------------------
			# Multivariate normal distribution
			# ----------------------------------
			k = x.size
			a = -0.5*np.dot(np.dot((x-u),np.linalg.inv(s)),(x-u))
			dets = np.linalg.det(s)

			rescale = 1.0/((np.sqrt((2.0*np.pi)**k*np.linalg.det(s))))

			return (1.0/((np.sqrt((2.0*np.pi)**k*dets)))*np.exp(a))/rescale

		

		res =   self.c[j,0] * mnd(self, obs, self.u[j,0], self.s[j,0,:] ) + \
			 	self.c[j,1] * mnd(self, obs, self.u[j,1], self.s[j,1,:] ) + \
			 	self.c[j,2] * mnd(self, obs, self.u[j,2], self.s[j,2,:] )
		if len(res.shape):
			raise RuntimeError("EmissionMatrix: get: mnd returned matrix, not vector: "+str(res.shape))
		return res
		



class Emission_matrix:

	def __init__(self,M,Or):

		R,D,T = Or.shape

		print "-> New emission matrix"
		print "Matrices, Rows, Columns",R,D,T

		self.Or = Or
		self.R = R 					# of samples of same word
		self.D = D 					# of elements in a feature vec.
		self.T = T 					# of feature vectors
		self.M = M 					# number of mixtures
		
		
		self.c = np.zeros((T,3))
		self.mu = np.zeros((T,3,D))
		self.Sigma = np.zeros((T,3,self.D,self.D))

		self.k_means_cluster()


	def multivar_normpdf(self,ot,mu_jk,Sigma_jk):
		
		exponent = 0.0
		coeff = 0.0
		result = 0.0

		exponent = -0.5*np.dot((ot-mu_jk),np.dot(np.linalg.inv(Sigma_jk),(ot-mu_jk)))
		coeff = 1.0/(6.283185307179586*(np.linalg.det(Sigma_jk)**0.5))
		result = coeff*math.exp(exponent)

		

		if DEBUG:
			print Sigma_jk
			print "Det",np.linalg.det(Sigma_jk)
			print str(coeff) + " exp( "+str(exponent) + ")"
		
		return result



	def get(self,j,ot):
		c = self.c
		mu = self.mu
		sigma = self.Sigma

		return  c[j,0]*self.multivar_normpdf(ot,mu[j,0],sigma[j,0]) + \
				c[j,1]*self.multivar_normpdf(ot,mu[j,1],sigma[j,1]) + \
				c[j,2]*self.multivar_normpdf(ot,mu[j,2],sigma[j,2])
		

	def proj(self,diag_arr):
		M= len(diag_arr)
		D = np.zeros((M,M))
		for i in xrange(0,M):
			if diag_arr[i] > 1e-3:
				D[i,i] = diag_arr[i]
			else:
				D[i,i] = 1e-3
		return D


	def k_means_cluster(self):
		print "-> K-means clustering"

		R,N,T = self.Or.shape


		for j in xrange(0,T):

			# Pick every sample of state j
			ot_j = np.zeros((R,N))
			for r in xrange(0,R):
				ot_j[r,:] = self.Or[r,:,j]
			
			centroids,_ = scipy.cluster.vq.kmeans( ot_j, self.M )
			idx,_  = scipy.cluster.vq.vq(ot_j,centroids)
			idx = list(idx)	

			attempts = 0
			while idx.count(0)<2 or idx.count(1)<2 or idx.count(2)<2 :

				centroids,_ = scipy.cluster.vq.kmeans( ot_j, self.M )
				idx,_  = scipy.cluster.vq.vq(ot_j,centroids)
				idx = list(idx)

				if attempts > 100:
					print "K-means did not converge, choosing randomly"
					idx = [0,0,1,2]
					break

				attempts += 1
			




			cluster0 = ot_j[[i for i in xrange(0,len(idx)) if idx[i]==0],:]
			cluster1 = ot_j[[i for i in xrange(0,len(idx)) if idx[i]==1],:]
			cluster2 = ot_j[[i for i in xrange(0,len(idx)) if idx[i]==2],:] 			

			if min(cluster0.shape)<2 or min(cluster1.shape)<2 + min(cluster2.shape)<2:
				print "Clustering failed, acquire more samples!"
				exit(1)

			num_fv = float(len(idx))

			self.c[j,0] = float(idx.count(0))/num_fv
			self.c[j,1] = float(idx.count(1))/num_fv
			self.c[j,2] = float(idx.count(2))/num_fv

			self.mu[j,0] = np.mean(cluster0,axis=0)
			self.mu[j,1] = np.mean(cluster1,axis=0)
			self.mu[j,2] = np.mean(cluster2,axis=0)

			
		
			# Should this be the entire matrix or the diagonal elements
			cov0 = np.diag(np.cov(cluster0.T,rowvar=1))
			cov1 = np.diag(np.cov(cluster1.T,rowvar=1))
			cov2 = np.diag(np.cov(cluster2.T,rowvar=1))

			self.Sigma[j,0] = self.proj(cov0)
			self.Sigma[j,1] = self.proj(cov1)
			self.Sigma[j,2] = self.proj(cov2)
			
		
			"""
			plt.subplot(211)
			plt.plot(cluster0.T)
			plt.title("Cluster 0")
			plt.subplot(212)
			plt.plot(np.diag(self.Sigma[j,0]).T)
			plt.title("Cluster mean")
			plt.show()
			"""
			


def compute_alpha(A,B,pi,O):
	
	N,T = O.shape

	alpha = np.zeros((T,N))

	# Initialization
	for i in xrange(0,T):
		#print B.get(i,O[0,:])
		alpha[0,i] += pi[i]*B.get(i,O[:,0])

	# Induction
	for t in xrange(0,T-1):

		for i in xrange(0,N):
			b = B.get(i,O[:,t+1])
			s = np.dot(alpha[t,:],A[:,i])
			#print alpha[t,:]
			alpha[t+1,i] = b*np.sum(s)

	
	# Termination
	#print alpha
	return alpha
	

def alpha_hat(alpha,t,i):
	alpha = forward_a(A,B,pi,O,tstop,state)
	return alpha[t,i]/sum(alpha[t])


def compute_beta(A,B,O):

	N,T = O.shape

	beta = np.zeros((T,N))

	# Initialization
	beta[-1,:] = 1.0

	# Induction
	for t in range(0,T-1)[::-1]:
		
		for i in range(0,N):
			beta_h = sum( [beta[t+1,j]*A[i,j]*B.get(j,O[:,t+1]) for j in xrange(0,N)] )
			beta[t,i] = beta_h
		
	# Termination
	
	return beta
	

def gamma(alpha,beta,B,Or,r,t,j,k):
	R,N,T = Or.shape
	num = 0.0
	denom = 0.0

	num = alpha[r,t,j]*beta[r,t,j]*B.c[j,k]*B.multivar_normpdf(Or[r,:,t],B.mu[j,k],B.Sigma[j,k])
	for k in xrange(0,B.M):
		denom += sum( [alpha[r,t,j]*beta[r,t,j] for j in xrange(0,T)] ) * \
				 sum([B.c[j,k2]*B.multivar_normpdf(Or[r,:,t],B.mu[j,k2],B.Sigma[j,k2]) for k2 in xrange(0,B.M)])

	return num/denom


def baum_welch(A,B,pi,Or):
	
	print "-> Baum-Welch parameter reestimation"
	R,N,T = Or.shape

	it = 0
	while it < 1:
		it += 1

		alpha = np.zeros((R,T,N))
		beta = np.zeros((R,T,N))
		for r in xrange(0,R):
			a = compute_alpha(A,B,pi,Or[r,:,:])
			b = compute_beta(A,B,Or[r,:,:])
			c = 1.0/np.sum(a,axis=1)

			for t in xrange(0,N):
				alpha[r,t,:] = c[t]*a[t,:]
				beta[r,t,:] = c[t]*b[t,:]

		
		
		print "Computing PI: "
		# PI reestimation		
		for i in xrange(0,T):

			print i,T

			num = 0.0
			denom = 0.0
			for r in xrange(0,R):
				num += alpha[r,1,i]*beta[r,1,i]
				denom += sum([alpha[r,T-1,i] for i in xrange(0,N)])

			pi[i] = num/denom

			

		#print "Pi",pi

		print "Computing A:"
		# A reestimation
		for i in xrange(0,N):
			print i,N
			for j in xrange(0,N):
				if A[i,j] > 1e-9:

					num = 0.0
					denom = 0.0
					for r in xrange(0,R):
						N,Tr = Or[r,:,:].shape
						num += sum([alpha[r,t,i]*A[i,j]*B.get(j,Or[r,t+1,:])*beta[r,t+1,j] for t in xrange(0,Tr-1)])
						denom += sum([alpha[r,t,i]*beta[r,t+1,j] for t in xrange(0,Tr-1)])

					A[i,j] = num/denom

		#print "A",A

		#print "C,mu,Sigma"
		print "Computing C"
		for j in xrange(0,T):
			for k in xrange(0,B.M):

				# C reestimation
				num = 0.0
				denom = 0.0
				for r in xrange(0,R):
					for t in xrange(0,T):

						num = gamma(alpha,beta,B,Or,r,t,j,k)
						for k2 in xrange(0,B.M):
							denom += gamma(alpha,beta,B,Or,r,t,j,k2)
				B.c[j,k] = num/denom



				# MU reestimation
				num = 0.0
				denom = 0.0
				#for r in xrange(0,R):
				#	for t in xrange(0,T):
				for r in xrange(0,R):
					for t in xrange(0,T):
						g = gamma(alpha,beta,B,Or,r,t,j,k)

						num += g*Or[r,:,t]
						denom += g
				B.mu[j,k] = np.mean(num)/denom

				# SIGMA reestimation
				num = 0.0
				denom = 0.0
				#for r in xrange(0,R):
				#	for t in xrange(0,T):
				for r in xrange(0,R):
					for t in xrange(0,T):
						g = gamma(alpha,beta,B,Or,r,t,j,k)
						
						num += g*np.dot((Or[r,:,t]-B.mu[j,k]).T,(Or[r,:,t]-B.mu[j,k]))

						denom += g

				
				B.Sigma[j,k] = num/denom
			print j,T

		#print "C",B.c
		#print "mu",B.mu
		#print "Sigma",B.Sigma

		#sys.stdout.write('.')
		#sys.stdout.flush()
		#it = it +1
		#print A
	
	return [A,B,pi]


class MultiObsSeq:
	"""
	One observation looks like:

		 o1 o2 o3 ... ot
		+---------------+
		|				|
		|				|
		|				| N
		|				|
		|				|
		|				|
		+---------------+
				M
		and is an NxM matrix

	All observations look like:

	[O1,O2,...OR]
	<------------>
		  R
	"""

	N = 0 # of columns <=> feature vector length (for one observation)
	M = 0 # of rows <=>  Number of observations
	R = 0 # of 

	PI = None
	A = None
	B = None

	observations = []
	matrix = None
	backlog = []

	def read_from_path(self,path):

		files = glob.glob(path+"/*.fv")

		if path == None:
			print "MultiObsSeq: read_from_file: You forgot to spec. the path"
			return
		
		for fname in files:
			print fname
			try:
				fobj = open(fname,"r")
				samples = np.asarray(pickle.load(fobj))
				fobj.close()
				self.add(samples)
			except IOError, e:
				print "MultiObsSeq: read_from_file: Unable to open file ",fname


		self.observations = np.matrix(self.observations)
		print "MultiObsSeq: Data dimensions (feature size, output symbols, recordings):",self.N,self.M,self.R


	def add(self, fv): 

		print fv.shape

		# Check if this is the first time we load a vector
		if self.N==0 and self.M==0 and self.R==0:
			N,D = fv.shape 
			self.N = N
			self.M = D
			self.R = 1
			self.observations = [fv]
		

		else:
			# ----------------------------------
			# Check dimensions
			# ----------------------------------
			N,D = fv.shape 
			if D != self.M:
				print "MultiObsSeq: add: fvdata has wrong dimensions"
				exit(1)

			# ----------------------------------
			# all requirements checked in, add to list
			# ----------------------------------
			self.R += 1
			self.observations.append(fv)
	

	def generate_matrix(self):
		print "MultiObsSeq: Generating a matrix from data"
		self.matrix = np.zeros((self.N,self.M,self.R)) # x y z
		z = 0
		for observationSeq in self.observations:
			self.matrix[0,:,z] = observationSeq
			z += 1

	def precompute(self):
		print "MultiObsSeq: generate_matrix: Precomputing values"
		O = self.matrix
		# ----------------------------------
		# Alpha( time t, state i/j )
		# ----------------------------------
		#self.alpha = np.zeros((self.N,self.M,self.R))
		print "@precompute: Warning self.alpha is MxMxR?"
		self.alpha = np.zeros((self.M,self.M,self.R))

		for r in range(0,self.R):
			t = 1
			for i in range(0,self.N):									# init
				self.alpha[0,i,r] = self.PI[i]*self.B.get(i,O[:,0,r])
			for t in range(0,self.M-1):									# induction
				for j in range(0,self.M-1):
					self.alpha[t+1,j,r] = self.B.get(j,O[:,t+1,r])*sum(np.multiply(self.alpha[t,:,r],self.A[j,:,r]))

		


	def train_model(self):
		print "MultiObsSeq: train_model: generating matrices and precomuting"
		
		# ----------------------------------
		# Initialize A with guess
		# ----------------------------------
		print "@train_model: Warning self.alpha is MxMxR?"
		self.A = np.zeros((self.M,self.M,self.R))
		for j in xrange(2,self.N):
			self.A[j-2,j-2] = 1.0/3.0
			self.A[j-2,j-1] = 1.0/3.0
			self.A[j-2,j  ] = 1.0/3.0
		self.A[self.N-2,self.N-2] = 0.5; self.A[self.N-2,self.N-1] = 0.5;
		self.A[self.N-1,self.N-1] = 1.0
		# ----------------------------------
		# Initialize PI with guess
		# ----------------------------------
		self.PI = np.zeros((self.N,1))
		self.PI[0] = 1.0
		# ----------------------------------
		# Initialize B parameters with guess's
		# ----------------------------------
		self.generate_matrix()
		self.B = EmissionMatrix(self.matrix)
		self.precompute()
		

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
		#print name,fv.shape
		self.observations.append((name,fv))

	def DTWdistance(self, w, x):

		#print "Comparing",w.shape,x.shape

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

		print sorted(results, key=lambda k: k[1])


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
		
		#if len(x1) < 1:
		#print "No voice detected"

		#	return
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

		#self.train_model(O)
		#B = Emission_matrix(3,O)
		#B.get(O[0])
	
	
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
		#sig = [float(e) for e in sig]/np.max(sig)



		return sig

	def load_feature_vectors(self, path="./storage/"):
		
		fv_files = glob.glob(path+"*.fv")
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

		#self.B = Emission_matrix(3,self.Or)
		#self.B = Emission_matrix(3,np.asarray(self.Or))
			

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


def print_header(ch):
	print "+"+"-"*(len(ch)+2)+"+"
	print "| "+ch+" |"
	print "+"+"-"*(len(ch)+2)+"+"

def test_emission_matrix():
	SR = SoundRecon()
	SR.load_feature_vectors()
	SR.B = Emission_matrix(3,SR.Or)

	ot = SR.B.Or[0,0]
	
	print_header("mu(mixture 1)-ot")
	print SR.B.mu[0,0]-ot
	print_header("mu(mixture 2)-ot")
	print SR.B.mu[0,1]-ot
	print_header("mu(mixture 3)-ot")
	print SR.B.mu[0,2]-ot

	print_header("norm(mu(M1)-ot")
	print np.linalg.norm(SR.B.mu[0,0]-ot)
	print_header("norm(mu(M2)-ot")
	print np.linalg.norm(SR.B.mu[0,1]-ot)
	print_header("norm(mu(M3)-ot")
	print np.linalg.norm(SR.B.mu[0,2]-ot)

	print_header("Sigma(M1) diagonals norm")
	print np.linalg.norm(np.diag(SR.B.Sigma[0,0]))
	print_header("Sigma(M2) diagonals norm")
	print np.linalg.norm(np.diag(SR.B.Sigma[0,1]))
	print_header("Sigma(M3) diagonals norm")
	print np.linalg.norm(np.diag(SR.B.Sigma[0,2]))



	print_header("Inverse Sigma(M1)")
	print np.linalg.norm(np.linalg.inv(SR.B.Sigma[0,0]))
	
	print_header("Inverse Sigma(M2)")
	print np.linalg.norm(np.linalg.inv(SR.B.Sigma[0,1]))
	
	print_header("Inverse Sigma(M3)")
	print np.linalg.norm(np.linalg.inv(SR.B.Sigma[0,2]))

	print SR.B.c

	print "="*40
	for ot in SR.B.Or[:,0,0]:
		print max([SR.B.get(j,ot) for j in range(0,20)])

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

	#plot_word_feature_vector("./storage/foo")
	#plot_word_feature_vector("./storage/bar")
	#plot_word_feature_vector("./storage/baz")

	#MOS = MultiObsSeq()
	#MOS.read_from_path("./storage")
	#MOS.train_model()

	
	

	SR = SoundRecon()
  	speech = SR.capture_word()
	obs = SR.audio_processor(speech)
	DTW = DynamicTimeWarper()
	DTW.read_from_path("./storage")
	#a = 0
	#b = len(DTW.observations)
	#c = int(random.uniform(a,b))
	#print "Testing:",DTW.observations[c][0]
	DTW.recognize( obs )



	#DTW.recognize(obs)


	# for i in range(0,10):
	# 	print "Ready?"
	# 	time.sleep(1)
	#   	speech = SR.capture_word()
	#   	SR.audio_processor(speech,"isis"+str(i)+".fv")

	#test_emission_matrix()



