/* Trabajo 2: Vision por Computador
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <cstdio>

using namespace cv;
using namespace std;
//VARIABLES GLOBALES
#define VERBOSO false
#define OTSU false

struct Muestra
{
	double momento0;
	double momento1;
	double momento2;
	double perimetro;
	int label;
};

int labelToInt(std::string label)
{
	if (label.compare("circulo") == 0) {
		return 1;
	}
	else if (label.compare("rectangulo") == 0) {
		return 2;
	}
	else if (label.compare("rueda") == 0) {
		return 3;
	}
	else if (label.compare("triangulo") == 0) {
		return 4;
	}
	else if (label.compare("vagon") == 0) {
		return 5;
	}
}
//FUNCIONES
/**
* Calcula el histograma de una imagen en blanco y negro.
*/
/**
* Muestra el histograma por pantalla de la imagen dada.
*/
void histogramaGris(Mat src, std::string nombre) {
	std::vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	/// Display
	namedWindow(nombre, CV_WINDOW_AUTOSIZE);
	imshow(nombre, histImage);

}

/**
 * Umbraliza la imagen
 */
void umbralizar(Mat imagen) {
	if (OTSU) {
		//Otsu
		threshold(imagen, imagen, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
	}
	else if (!OTSU) {
		//Adaptativo
		adaptiveThreshold(imagen, imagen, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 87, 2);
	}
}

//MAIN
int main2(int argc, char** argv)
{
	Mat imagen, destino;
	int * histograma;
	int recorte = 5;
	vector<vector<Point> > contours;
	RNG rng(12345);
	
	imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	if (VERBOSO) imshow("Original", imagen);

	imagen = imagen(cv::Rect (recorte, recorte, imagen.cols-recorte-1, imagen.rows-recorte-1));

	if (VERBOSO) {
		imshow("Original", imagen);
		histogramaGris(imagen, "Histo");
	}

	umbralizar(imagen);
	
	findContours(imagen, contours, RETR_EXTERNAL,CHAIN_APPROX_NONE);
	/// Get the moments
	vector<Moments> mu(contours.size());
	cout << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
		
	}
	if (VERBOSO){
	///  Get the mass centers:
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		/// Draw contours
		Mat drawing = Mat::zeros(imagen.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8);
			circle(drawing, mc[i], 4, color, -1, 8, 0);
		}

		imshow("Umbralizada", imagen);
		imshow("Contours", drawing);
		cout << "m0: " << mu[0].m00 << " m1: " << (mu[0].mu20+ mu[0].mu02) << " m2: " << (mu[0].mu20 + mu[0].mu02)*(mu[0].mu20 + mu[0].mu02) + 4 * mu[0].mu11*mu[0].mu11 << " perimetro: " << arcLength(contours[0], true) << endl;

		cout << "Numero de figuras: " << contours.size() << endl;
		while (true)
			if (waitKey(10) == 27) break; //Para con la tecla escape
	}
	else {
		if (contours.size() > 1) {
			cout << "WARNING: mas de un contorno encontrado en entrenamiento" << endl;
			//return -1;
		}
		FILE * pFile;
		pFile = fopen("objetos", "a");

		for (int i = 0; i < contours.size(); i++) {
			if (arcLength(contours[i], true) > 5) fprintf(pFile, "%f %f %f %f \r", mu[i].m00, mu[i].mu20 + mu[i].mu02, (mu[i].mu20 + mu[i].mu02)*(mu[i].mu20 + mu[i].mu02)+4*mu[i].mu11*mu[i].mu11, arcLength(contours[i], true));
		}
	}

}