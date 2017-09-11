
#include <string>
#include "ModelCalibration.h"


int main(int argc, char const *argv[]) 
{
	string datapath = "C:\\Users\\hgudmund\\Documents\\SublimeProjects\\OptionData.txt";
	string outputpath = "output.txt";
	ModelCalibration cm; 
	cm.DataRetriever(datapath);
	cm.Calibrate();

	int input;
	cin >> input;
	return 0;
}
