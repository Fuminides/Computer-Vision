// ConsoleApplication3.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

#define PROFUNDIDAD 5
using namespace cv;

const bool ECUALIZAR_GREY =  false, ECUALIZAR_COLOR = false, ALIEN = false, POSTER = false, GHOSTING = true;

//Para el alien
double proporciones_piel[] = { 255.0/219.0, 255.0/172.0, 219.0/172.0 }; //R/G, R/B, G/B
double tolerancias[] = { 0.3, 0.7, 0.2};

//Para el poster
int numero_colores = 230;

//Para ghosting
bool guarda = false;
double alfa = 0.1;
int frames_acumulados = 0;

int main(int argc, char** argv)
{
	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.

	if (!cap.open(0))
		return 0;
	bool recorrer_matriz = false;

	if (ALIEN | POSTER) {
		recorrer_matriz = true;
	}

	Mat frameAnterior[PROFUNDIDAD];

	while (true)
	{
		Mat frame, frameOriginal;
		std::vector<Mat> channels;

		cap >> frame;
		frameOriginal = frame.clone();
		

		if (recorrer_matriz) {
			for (int i = 0; i < frame.rows; i++) {
				for (int z = 0; z < frame.cols; z++) {
					Vec3b canales = frame.at<Vec3b>(i, z);
					unsigned char G = canales[1], R = canales[2], B = canales[0];

					if (ALIEN) {
						double proporcionRG = R*1.0 / (G*1.0), proporcionRB = R*1.0 / (B*1.0),
							proporcionGB = G*1.0 / (B*1.0);
						//std::cout << "Valores: " << std::to_string(proporcionRG) << " " << std::to_string(abs(proporcionRG - proporciones_piel[1])) << std::endl;
						if ((abs(proporcionRG - proporciones_piel[0]) < tolerancias[0]) &&
							(abs(proporcionRB - proporciones_piel[1]) < tolerancias[1]) &&
							(abs(proporcionGB - proporciones_piel[2]) < tolerancias[2]) &&
							(abs(R - B) > 10) ) {
							//std::cout << "Entro" << std::endl;
							canales[0] = 0;
							canales[1] = 0;
							canales[2] = 255;
						}
						frame.at<Vec3b>(i, z) = canales;
					}

					if (POSTER) {
						canales[0] = B % numero_colores *1.0 / numero_colores * 255;
						canales[1] = G % numero_colores *1.0 / numero_colores * 255;
						canales[2] = R % numero_colores *1.0 / numero_colores * 255;
						frame.at<Vec3b>(i, z) = canales;
					}

				}
			}
		}

		if (GHOSTING) {
			if (guarda) {
				frameAnterior[PROFUNDIDAD - 1] = frameAnterior[PROFUNDIDAD - 1] * (alfa / PROFUNDIDAD);
				for (int i = 0; i < PROFUNDIDAD-1; i++) {
					frameAnterior[PROFUNDIDAD-1] = 
				}
				frame = frame*(1 - alfa) + frameAnterior*alfa;
			}
			else guarda = true;
			frameAnterior = frame.clone();
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
