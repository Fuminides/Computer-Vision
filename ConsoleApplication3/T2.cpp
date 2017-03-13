/* Trabajo 2: Vision por Computador
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
//VARIABLES GLOBALES
#define VERBOSO true
#define ADAPTATIVE true

struct Muestra
{
	double momento0;
	double momento1;
	double momento2;
	double perimetro;
};
//FUNCIONES
/**
* Calcula el histograma de una imagen en blanco y negro.
*/
int * histogramaGrisManualCalc(Mat src) {
	static int resultado[255];
	for (int i = 0; i < 255; i++) {
		resultado[i] = 0;
	}
	for (int i = 0; i < src.rows; i++) {
		for (int z = 0; z < src.cols; z++) {
			int indice = src.at<uchar>(i, z);
			//cout << "Ind: " << to_string(indice) << endl;
			resultado[indice] = resultado[indice] + 1;
		}
	}

	return resultado;
	
}
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
 * Devuelve el numero de picos presentes en el histograma dado.
 */
int numeroPicos(int * histograma_gris) {
	int numero_picos = 0, anterior = 0, actual;
	bool creciendo = false;

	for (int i = 0; i < 255; i++) {
		actual = histograma_gris[i];
		if (creciendo) {
			if (actual < anterior) {
				creciendo = false;
				numero_picos++;
			}
		}
		else {
			if (actual > (anterior+1)*2) {
				creciendo = true;
			}
		}
		anterior = actual;
	}

	return numero_picos;
}

//MAIN
int main(int argc, char** argv)
{
	Mat imagen, destino;
	int * histograma;
	int recorte = 3;
	vector<vector<Point> > contours;
	RNG rng(12345);

	imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	if (VERBOSO) imshow("Original", imagen);

	imagen = imagen(cv::Rect (recorte, recorte, imagen.cols-recorte-1, imagen.rows-recorte-1));

	if (VERBOSO) {
		imshow("Original", imagen);
		histogramaGris(imagen, "Histo");
	}

	if (ADAPTATIVE) {
		//Otsu
		threshold(imagen,imagen, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
	}
	else if (!ADAPTATIVE) {
		//Adaptativo
		adaptiveThreshold(imagen, imagen, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,87,2);
	}
	
	findContours(imagen, contours, RETR_EXTERNAL,CHAIN_APPROX_NONE);
	/// Get the moments
	vector<Moments> mu(contours.size());
	cout << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

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

	/// Show in a window
	if (VERBOSO) {
		imshow("Umbralizada", imagen);
		imshow("Contours", drawing);

		while (true)
			if (waitKey(10) == 27) break; //Para con la tecla escape
	}
	else {
		if (contours.size() > 1) cout << "WARNING: mas de un contorno encontrado" << "\n";

		for (int i = 0; i < contours.size(); i++) {
			Muestra muestra;
			muestra.momento0 = mu[i].m00;
			muestra.momento1 = mu[i].m01;
			muestra.momento2 = mu[i].m02;
			muestra.perimetro = arcLength(contours[i], true);
		}
	}

}