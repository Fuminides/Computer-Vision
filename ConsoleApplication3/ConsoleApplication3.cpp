// ConsoleApplication3.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
using namespace cv;

const bool ECUALIZAR_GREY =  false, ECUALIZAR_COLOR = true, ALIEN = true, POSTER = false;
unsigned char valores_piel[] = { 134, 192, 234 }; //Solo para piel caucásica
int tolerancias[] = { 55, 25, 55};

int main(int argc, char** argv)
{
	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.

	if (!cap.open(0))
		return 0;
	bool recorrer_matriz = false;

	if (ALIEN) {
		recorrer_matriz = true;
	}
	for (;;)
	{
		Mat frame, frameOriginal;
		std::vector<Mat> channels;

		cap >> frame;
		frameOriginal = frame.clone();

		if (recorrer_matriz) {
			for (int i = 0; i < frame.rows; i++) {
				for (int z = 0; z < frame.cols; z++) {
					if (ALIEN) {
						Vec3b canales = frame.at<Vec3b>(i, z);
						unsigned char G = canales[1], R = canales[2], B = canales[0];
						//std::cout << "Valores: " << std::to_string(G) << " " << std::to_string(abs(G - valores_piel[1])) << std::endl;
						if ((abs(G - valores_piel[1]) < tolerancias[1]) &&
							(abs(R - valores_piel[2]) < tolerancias[2]) &&
							(abs(B - valores_piel[0]) < tolerancias[0])) {
							//std::cout << "Entro" << std::endl;
							canales[0] = 0;
							canales[1] = 0;
							canales[2] = 255;
						}
						frame.at<Vec3b>(i, z) = canales;
					}
				}
			}
		}

		if (ECUALIZAR_GREY) {
			cvtColor(frame, frame, CV_BGR2GRAY);
			frameOriginal = frame.clone();
			equalizeHist(frame, frame);
		}
		else if (ECUALIZAR_COLOR) {
			cvtColor(frame, frame, CV_BGR2YCrCb);
			split(frame, channels);
			equalizeHist(channels[0], channels[0]);
			merge(channels, frame);
			cvtColor(frame, frame, CV_YCrCb2BGR);
		}
		
		if (frame.empty()) break; // end of video stream
		imshow("Modificada", frame);
		imshow("Original", frameOriginal);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	// the camera will be closed automatically upon exit
	// cap.close();
	return 0;
}
