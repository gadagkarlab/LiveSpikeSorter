#ifndef INPUT_GUI_H_
#define INPUT_GUI_H_

#include "../Networking/inputParameters.h"

#define _LOCAL_HOST "129.236.161.150" // modified this to defauly when i rebuild 
#define _DEFAULT_MASTER_PORT 8888

// TODO: REMOVE THIS DEPENDENECY, because weird stuff happens when I try to... everything crashes and nothing runs
// when I change this to something else, and this weird folder will show up as the default for one of the inputs
//"D:
#define _FILE_PATH "D:\\Rose62_d1_night_g0\\"
//"C:\\SGL_DATA\\01_27_p1_templategeneration_g0\\01_27_p1_templategeneration_g0_imec0\\"
#define _FOLDER "Rose62_d1_night_g0_imec0\\"
//"Lime59_d1_train\\"
//#define _FILE_PATH "Z:\GadagkarLab\Keshav\Code\livespikesorter\data"
//#define _FOLDER "Lime59_d1_train\\"
class InputGUI {
private:
	// InputGUI parameters
	int computerJob;

	// Used by both Master GUI and Sorter GUIs
	uint16 masterPort;

	// Used by Sorter GUIs
	std::string masterHost;

	// Used by Master GUI
	InputParameters Params;

	// For GPU calculator
	int iCCalc;
	int iLCalc;
	int iMCalc;

	// Filters for fileDialogs
	const char * binFilterPattern[1] = { "*.bin" };
	const char * txtFilterPattern[1] = { "*.txt" };

	// Methods
	void InputTextWithFileDialog(const char* label, std::string* str, const char* buttonLabel, const char *initDirectory, int numFilterPatterns, const char *filterPatterns[], int allowMultipleSelect, bool folderSelect);
	std::string getDeviceInfo(int deviceNumber);
	void gatherNetworkParameters();
	void gatherDataAccquisitionParameters();
	void gatherGPUParameters();
	void gatherSpikeSorterParameters();
	void gatherThresholdCrossingParameters();
	void gatherSorterParameters();
	void gatherParallelizedOSSInputs();
	void gatherDecoderParameters();
	// BRIAN
	void gatherFeedbackParameters();
	void writeParamFile();
	//
	double CalcBytes(long L, long C, long M, long MaxWin, long MinWin, long SpikesExpected);


public:
	InputGUI(InputParameters cmdLineParams);
	~InputGUI();

	void gatherInputParameters(bool &finished, bool &isNetworking);

	// Getters
	InputParameters getInputParameters() { return Params; };
	int &getComputerJob() { return computerJob; }
	std::string &getMasterHost() { return masterHost; }
	uint16 &getMasterPort() { return masterPort; }
};

#endif