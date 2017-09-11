#include <fstream>
#include <string>
#include <sstream>
#include "ParallelizedObjective.h"

class ModelCalibration
{
public:
	void DataRetriever(string datapath)
	{

		ifstream infile(datapath);
		int count = 0;
		while (infile && count < ReadLimit)
		{
			count += 1;
			string s;
			if (!getline(infile, s))
				break;

			istringstream sts(s);
			vector <string> record;

			while (sts)
			{
				string s;
				if (!getline(sts, s, ','))
					break;
				record.push_back(s);
			}

			Data.push_back(record);
		}
		if (!infile.eof())
		{
			cerr << "End of the line, brah\n";
		}

		cout << "size of data " << Data.size();
	}

	void Calibrate()
	{
		ParallelizedObjective parobj;
		for (int dayindex = 0; dayindex < ReadLimit; dayindex += NofOpts*NofStrikes + 1)
		{
			cout << "dayindex " << dayindex << endl;
			int nrows = Data.size();
			int ncolumns = Data[0].size();
			cout << "data size " << Data.size() << endl;
			vector<vector<Ty>> numerical(nrows, vector<Ty>(ncolumns, 0));

			for (int i = 0; i < NofOpts*NofStrikes; i++)
			{

				for (int j = 0; j < 9; j++) // seven fields in each record (with spotvols)
				{
					numerical[i][j] = atof(Data[i + dayindex][j].c_str());
					cout << Data[i + dayindex][j] << " ";
				}
				cout << endl;
			}

			OptionData ODList[NofOpts];
			int opcount = 0;
			for (int op = 0; op < NofOpts*NofStrikes; op += NofStrikes)
			{
				ODList[opcount].ap = numerical[op][0];
				ODList[opcount].rf = numerical[op][2];
				ODList[opcount].mat = numerical[op][1];
				ODList[opcount].spotvol = numerical[op][6];
				for (int k = 0; k < NofStrikes; k++)
				{
					ODList[opcount].SL[k] = numerical[op + k][3];
					ODList[opcount].OT[k] = numerical[op + k][4];
					ODList[opcount].MPL[k] = numerical[op + k][5];
					ODList[opcount].BIP[k] = numerical[op + k][7];
					ODList[opcount].ASP[k] = numerical[op + k][8];
				}
				opcount++;
			}

			cout << "options info " << ODList[0].ap << " " << ODList[1].mat << " " << ODList[1].rf << endl;
			parobj.Initialize(ODList);
			Vector<double> y(5);
			y << 2.22753, 0.0442048, 0.0312627, 1.39501, -0.833071;
			BfgsSolver<double> solver;
			//solver.minimize(parobj, y);
			//cout << "Parameter Estimate:      " << y.transpose() << std::endl;
			vector<Ty> xinit = { 2.22753, 0.0442048, 0.0312627, 1.39501, -0.833071 };
			parobj.PricePrintout(xinit);
			int input;
			cin >> input;
		}
	}

	void EstimatesToFile(string outputpath, vector<Ty> estimates)
	{
		ofstream myfile(outputpath);
		if (myfile.is_open())
		{
			for (auto xit = estimates.begin(); xit < estimates.end(); xit++)
				myfile << *xit << ",   ";
			myfile << "\n";
			myfile.close();
		}
	}
private:
	vector <vector <string>> Data;
};