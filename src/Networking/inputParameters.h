#ifndef INPUTPARAMETERS_H
#define INPUTPARAMETERS_H

typedef unsigned short  uint16;

#include <vector>
#include <map>
#include <unordered_set> // BRIAN

struct InputParameters {
	std::string							sInputFolder{},
										sImecFile{},
										sNidqFile{},
										sSpikesFile{},
										sLogFile{},
										sEventFile{},
										sDataAccquisitionHost{},
										sDecoderWorkFolder{},
										sDecoderInputFolder{},
										sOSSOutputFolder{},
										sSylNum{}, // BRIAN
										sTemplateIdx{}, // BRIAN
										sdmIP{};

	uint16								uDataAccquisitionPort,
										uSelectedDevice;

	std::vector<uint16>					vSelectedDevices,
										vChannelSubset;

	std::map<uint16, std::string>		mapDeviceFilePaths,
										mapOSSOutputFolders,
										mapDecoderInputFolders,
										mapSpikeFiles;

	double								dTau,
										dThreshold,
										dRatioToMax;

	float								fImecSamplingRate,
										fNidqSamplingRate,
										fDelay1, // BRIAN
										fDelay2, // BRIAN
										fDelay3, // BRIAN
										fPulseWindow, // BRIAN
										fThresholdStd;

	int									iSubstream,
										iNidqRefreshRate,
										iMinScanWindow,
										iMaxScanWindow,
										iConvolutionTimes,
										iDownsampling,
										iMaxIts,
										iTimeBehind,
										iAvgWindowTime,
										iRedundancy,
										iWindowLength,
										iBinLength,
										iWindowOffset,
										iSorterType,
										iDigLineIdx, // BRIAN
										iNumAnChans, // BRIAN
										iThresh, // BRIAN
										iNumTemplates;

	bool								bReadFromFile,
										bIsDecoding,
										bIsSendingFeedback,
										bFeedbackMode, // BRIAN
										bThreshMode, // BRIAN
										bSmallskip;

	std::vector<int>					vSylNum; // BRIAN

	std::unordered_set<int>				usTemplateIdx; // BRIAN

	uint16_t							sdmPort{};
};
#endif