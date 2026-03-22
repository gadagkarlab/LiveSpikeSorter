#if _WIN32 || _WIN64
#include <windows.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

#include <sstream>
#include <errno.h>
#include <iostream>
#include <cmath>
#include <bitset>
#include <atomic>
#include "../../External/sglx/SglxCppClient.h"

#include "dataSocket.h"
#include "../Helpers/utils.h"

#include "../Helpers/Timer.h"

static std::mutex sglxMutex;

// ------------------------------------------------------------------------------
//
// Name			: StreamDataSocket::StreamDataSocket()
//
// Description  : Constructor, initialize retrieve buffer and connection settings
//
// ------------------------------------------------------------------------------
StreamDataSocket::StreamDataSocket(std::string accquisitionHost, uint16 accquisitionPort, int substream, long lMaxSize, long lMinSize, float fImecSampRate, float fNiqdSampRate, int downSampling)
	: m_sHost(accquisitionHost)
	, m_uPort(accquisitionPort)
	, m_iSubstream(substream)
	//, m_vImecChannels(channelMap, channelMap + lNChans) // Fill vector with values of *int array
	// KS - NI DIG LINE IS HARDCODED HERE for me it should be 3rd bc i have 2 analog chans, 0 indexed !
	, m_vNidqChannels({ 2 }) //# analog chans + 1 
{
	static const char *ptLabel = { "StreamDataSocket::StreamDataSocket" };

	// Initialize the static data members inherited by parent DataSocket class
	m_lMaxSize = lMaxSize;
	m_lMinSize = lMinSize;
	//m_lNChans = lNChans;
	m_lDownsampling = 1; // = downSampling;
	m_fImecSampRate = fImecSampRate;
	m_fNidqSampRate = fNiqdSampRate;

	if (!connect()) {
		std::cout << "Error while trying to establish connection to " << accquisitionHost << ":" << accquisitionPort << "!" << std::endl; // TODO redundant amount of errors. Also need more errors for other parts
		_RUN_ERROR(ptLabel, "Couldn't establish connection");
	}
};
StreamDataSocket::~StreamDataSocket() {
	disconnect();
};


bool StreamDataSocket::connect() {

	//m_mSGlxMutex.lock();
	std::cout << "Trying to connect to " << getHost() << ":" << getPort() << "..." << std::endl;
	{
		std::unique_lock<std::mutex> lock(m_mSGlxMutex);
		if (!sglx_connect_std(S, m_sHost, m_uPort)) {
		error:
			printf("error [%s]\n", S.err.c_str());
		}
	}
	std::cout << "...Connection successful!" << std::endl;
	//m_mSGlxMutex.unlock();
	return true;
}


void StreamDataSocket::disconnect() {
	m_mSGlxMutex.lock();
	sglx_close(S);
	std::cout << "... Disconnected!" << std::endl;
	m_mSGlxMutex.unlock();
}

t_ull StreamDataSocket::getStreamSampleCt(int streamType, OSSSpecificParams osParams) {
	//m_mSGlxMutex.lock();
	std::unique_lock<std::mutex> lock(sglxMutex);
	t_ull sampleCt = (streamType == IMEC) ? sglx_getStreamSampleCount(S, streamType, osParams.substream) : sglx_getStreamSampleCount(S, streamType, 0);
	//m_mSGlxMutex.unlock();
	return sampleCt;
}

float StreamDataSocket::getStreamSampleRate(int streamType, OSSSpecificParams osParams) {
	//m_mSGlxMutex.lock();
	std::unique_lock<std::mutex> lock(sglxMutex);
	float sampleCt = (streamType == IMEC) ? sglx_getStreamSampleRate(S, streamType, osParams.substream) : sglx_getStreamSampleRate(S, streamType, 0);
	//m_mSGlxMutex.unlock();
	return sampleCt;
}



// sglx_fetch returns the sample count index of first sample in matrix, or zero if error. StreamDataSocket::fetch, however,
// returns the sample count index of the last sample in matrix.
t_ull StreamDataSocket::fetch(std::vector<short> &data, t_sglxconn &S, int streamType, int substream, t_ull startSamp, int maxSamps, const std::vector<int> &channelSubset) {
	//m_mSGlxMutex.lock();
	std::unique_lock<std::mutex> lock(sglxMutex);
	t_ull lLatestCt = sglx_fetch(data, S, streamType, substream, startSamp, maxSamps, channelSubset) + maxSamps;
	//m_mSGlxMutex.unlock();
	return lLatestCt;
}

// fetchLatest: lStreamSampleCt (the start of the fetch) can change
t_ull StreamDataSocket::fetchLatest(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	Timer timer("fetchLatest()");
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// If default start value, fetch the min size. Otherwise, calculate the number of samples from lStartCt to current
	t_ull lToGet = lLatestCt - lStartCt;
	if (lStartCt == ULLONG_MAX) {
		lToGet = m_lMinSize;
	}

	// Limit the number of samples fetched to lMaxSize samples
	if (lToGet > m_lMaxSize) {
		lToGet = m_lMaxSize;
	}

	lStartCt = lLatestCt - lToGet;
	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill fData with m_sFetchBuffer's contents
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++) {
		fData[lI] = (float)m_sFetchBuffer[lI];
	}
	return lLatestCt;
}

t_ull StreamDataSocket::fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// If default start value, fetch the min size. Otherwise, calculate the number of samples from lStartCt to current
	t_ull lToGet = lLatestCt - lStartCt;
	if (lStartCt == ULLONG_MAX) {
		lToGet = m_lMinSize;
		lStartCt = lLatestCt - m_lMinSize;
	}

	// Limit the number of samples fetched to lMaxSize samples
	if (lToGet > m_lMaxSize) {
		lToGet = m_lMaxSize;
		lStartCt = lLatestCt - m_lMaxSize;
	}

	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill fData with m_sFetchBuffer's contents
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++)
		fData[lI] = (float)m_sFetchBuffer[lI];

	//std::cout << "number of samples: " << lToGet << std::endl;
	//std::cout << "number of channels: " << m_lNChans << std::endl;

	return lLatestCt;
}

// fetchFromPlace: lStreamSampleCt (the start of the fetch) is constant
t_ull StreamDataSocket::fetchFromPlace(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// Limit the fetch amount to the max size
	t_ull lToGet = min(m_lMaxSize, lLatestCt - lStartCt);

	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill data buffer with m_sFetchBuffer
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++)
		fData[lI] = (float)m_sFetchBuffer[lI];

	return lLatestCt;
}

//KS fxn - trying to get IMEC buffer with exact times for sorting 
t_ull StreamDataSocket::fetchImecExact(float *fData, OSSSpecificParams osParams, t_ull lStartCt, t_ull lEndCt)
{
	t_ull lToGet = lEndCt - lStartCt;

	t_ull lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill data buffer with m_sFetchBuffer
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++)
		fData[lI] = (float)m_sFetchBuffer[lI];

	return lLatestCt;
}

t_ull StreamDataSocket::initNidqStream() {
	/* Wait some time before starting NIDQ stream fetching to avoid SpikeGLX errors.
	If you are receiving "[FETCH: Too late.]" error messages from
	the fetchEventInfo call, increase the time waited. */
	Sleep(100); // TODO explore how low this value can go.
	//commented out for invalid argument
	//t_ull lLatestSampleCt = getStreamSampleCt(NIDQ);
	//re
	return 1;
}



//KS made based on StreamDataSocket::fetchLatest
t_ull StreamDataSocket::fetchNidqLatest(float *fData_NI, OSSSpecificParams osParams, t_ull lStartCt, int m_nMaxSize, int m_nMinSize) {
	//just copied from fetchLatest need to figure out how to get the right bit for my digline. this assumes i will never fall behind 
	
	t_ull lLatestCt = getStreamSampleCt(NIDQ, osParams);
	
	t_ull lToGet = lLatestCt - lStartCt;
	if (lStartCt == ULLONG_MAX) {
		lToGet = m_nMinSize;
	}
	if (lToGet > m_nMaxSize) {
		lToGet = m_nMaxSize;
	}
	lStartCt = lLatestCt - lToGet;
	lLatestCt = fetch(m_sNidqBuffer, S, NIDQ, 0, lStartCt, lToGet, m_vNidqChannels);// hard code in digital params 

	constexpr int bitIdx = 2;// hardcoded line i expect my syll code to come on 
	//Fill fData with m_sNidqBuffer's contents after selecting my digital word buffer should b ok 
	for (t_ull lI = 0; lI < lToGet; ++lI) {
		int bitVal = (m_sNidqBuffer[lI] >> bitIdx) & 1;
		fData_NI[lI] = static_cast<float>(bitVal);
		//if (bitVal==1) { std::cout << "Gotcha!" << std::endl; }
	}
	return lLatestCt;
}


t_ull StreamDataSocket::fetchNidqLatestAndCountEdges(OSSSpecificParams osParams, t_ull lStartCt, bool &prevHigh, int &edgeCount, std::vector<t_ull> &edgeTimes, int m_nMaxSize, int m_nMinSize) {
	
	constexpr int bitIdx = 2;

	t_ull lLatestCt = getStreamSampleCt(NIDQ, osParams);

	t_ull lToGet = lLatestCt - lStartCt;

	if (lStartCt == ULLONG_MAX) {
		lToGet = m_nMinSize;
	}
	if (lToGet > m_nMaxSize) {
		lToGet = m_nMaxSize;
	}

	lStartCt = lLatestCt - lToGet;

	lLatestCt = fetch(m_sNidqBuffer, S, NIDQ, 0, lStartCt, lToGet, m_vNidqChannels);

	edgeCount = 0;
	edgeTimes.clear();
	
	for (t_ull i = 0; i < lToGet; ++i) {
		bool high = ((m_sNidqBuffer[i] >> bitIdx) & 1) & 1;

		if (!prevHigh && high) {
			++edgeCount;
			edgeTimes.push_back(lStartCt + i);
		}

		prevHigh = high;
	}

	return lLatestCt;
}

// fetch the NIDQ data and extract stimulus event time and label (if they exist)
t_ull StreamDataSocket::fetchEventInfo(int &eventLabel, t_ull lStartCt, OSSSpecificParams osParams) {
	// Compute how many samples needed to get to present time
	//lToGet is always zero

	efficientWait(20);
	t_ull lToGet = getStreamSampleCt(NIDQ, osParams) - lStartCt;
	lToGet = getStreamSampleCt(NIDQ, osParams) - lStartCt;
	//std::cout << lToGet << "\n";
	// Fetch NIDQ data (substream is 0 by convention)

	//m_mSGlxMutex.lock();
	t_ull lLatestCt;
	{
		std::unique_lock<std::mutex> lock(sglxMutex);
		lLatestCt = sglx_fetchLatest(m_sNidqBuffer, S, NIDQ, 0, lToGet, m_vNidqChannels) + lToGet;
	}
	//m_mSGlxMutex.unlock();
	// Search the data for a non zero and break out if found, identifying the stimulus event label

	for (int i = 0; i < m_sNidqBuffer.size(); i++) {
		if (m_sNidqBuffer[i] == 2 || m_sNidqBuffer[i] == 4) {
			eventLabel = m_sNidqBuffer[i];
			std::cout << eventLabel;
			efficientWait(100);
			//std::cout << "TRIAL!";

			break;

		}
	}

	return lLatestCt;
}

//fetch & translate digital syllable code 

//KS needs to check the lines I use and make sure it works. this uses a much older form of the API and im moderately concerened neweer SGLx wont support it
void StreamDataSocket::setDigitalOut(int signal) {

	constexpr const char* LINE5 = "PXI1Slot4_2/port0/line5";
	constexpr const char* LINE7 = "PXI1Slot4_2/port0/line7";

	if (signal == 0) {
		//Sleep(1);
		sglx_setDigitalOut(S, 1, LINE5);//dig line is hardcoded here 
		//efficientWait(1);
		sglx_setDigitalOut(S, 0, LINE5);
	}
	if (signal == 1) {
		sglx_setDigitalOut(S, 1, LINE7);//dig line is hardcoded here 
		//efficientWait(1);
		sglx_setDigitalOut(S, 0, LINE7);
	}

	/*std::string binaryFeedbackSignal = std::bitset<nBits>(signal).to_string();
	std::cout << binaryFeedbackSignal<<"\n";
	efficientWait(1000);
	for (int i = 0; i < nBits; i++) {
		hiLo = binaryFeedbackSignal[i] == '1' ? true : false;
		line = "PXI1Slot2/port1/line" + std::to_string(i + 1); // TODO It appeared that the line was 1 indexed (why there's a plus 1), but double check
		sglx_setDigitalOut(S, hiLo, "PXI1Slot2/port1/line1");


	}*/

}



bool StreamDataSocket::isRunning() {
	bool running;

	std::unique_lock<std::mutex> lock(sglxMutex);
	sglx_isRunning(running, S);
	return running;
}


bool StreamDataSocket::startRun() {
	if (!isRunning()) {
		///m_mSGlxMutex.lock();
		{
			std::unique_lock<std::mutex> lock(sglxMutex);
			sglx_startRun(S);
		}
		//m_mSGlxMutex.unlock();
		std::cout << "New run started..." << std::endl;

		Sleep(5000); // Wait 5s for startup
	}
	else
		std::cout << "... run already started." << std::endl;
	return true;
}


bool StreamDataSocket::stopRun() {
	if (isRunning()) {
		//m_mSGlxMutex.lock();
		{
			std::unique_lock<std::mutex> lock(sglxMutex);
			sglx_stopRun(S);
		}
		//m_mSGlxMutex.unlock();
		std::cout << "Run stopped..." << std::endl;
	}
	else
		std::cout << "... no run started." << std::endl;
	return true;
}


void StreamDataSocket::waitUntil(t_ull lWaitUntilCt, OSSSpecificParams osParams) {

	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);
	if (lLatestCt < lWaitUntilCt) {
		int iWaitTime = (lWaitUntilCt - lLatestCt) / (m_fImecSampRate / 1000); // ms
		efficientWait(iWaitTime);
	}
}


void StreamDataSocket::waitUntilIMEC(t_ull lWaitUntilCt, float ImSampRate, OSSSpecificParams osParams){
	
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);
	if (lLatestCt < lWaitUntilCt) {
		int iWaitTime = (lWaitUntilCt - lLatestCt) / (ImSampRate / 1000); // ms
		efficientWait(iWaitTime);
	}
}
void StreamDataSocket::waituntilNI(t_ull lWaitUntilCt, float NiSampRate, OSSSpecificParams osParams){
	
	t_ull lLatestCt = getStreamSampleCt(NIDQ, osParams);
	if (lLatestCt < lWaitUntilCt) {
		int iWaitTime = (lWaitUntilCt - lLatestCt) / (NiSampRate / 1000); // ms
		efficientWait(iWaitTime);
	}
}
