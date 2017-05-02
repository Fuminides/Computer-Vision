#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Windows.h>

using namespace cv;
using namespace std;

#define USE_BRISK 0 //Puntos BRISK + alg. BRISK
#define USE_ORB 1 //Puntos FAST + alg. ORB

#define USE_BRUTE 3
#define USE_VECINO 4
#define USE_FLANN 5

#define TAM_MODELO 4


/**
 * Calcula los keypoints de una imagen.
 */
vector<KeyPoint> compute_keypoints(Mat imagen, int mode) {
	Ptr<FeatureDetector> detector;
	switch (mode) {
	case USE_ORB:
		detector = FastFeatureDetector::create(10);
		break;
	case USE_BRISK:
		detector = BRISK::create();
		break;
	}

	vector<KeyPoint> keypoints;
	detector->detect(imagen, keypoints);

	return keypoints;
}

/**
 * Extreae los descriptores de los keypoints de una imagen.
 */
Mat extraer_descriptores(Mat imagen, vector<KeyPoint> keypoints, int mode) {
	Mat descriptors;
	Ptr<DescriptorExtractor> descriptorExtractor;

	switch (mode) {
	case USE_BRISK:
		descriptorExtractor = BRISK::create();
		descriptorExtractor->compute(imagen, keypoints, descriptors);

		return descriptors;
	case USE_ORB:
		Ptr<ORB> orb = ORB::create();
		orb->compute(imagen, keypoints, descriptors);

		return descriptors;
	}
	
}

/**
 * Calcula los matches entre distintos keypoints de dos imagenes a partir de sus descriptores.
 *
 * Permite utilizar busqueda por fuerza bruta con minimo de similitud, por fuerza bruta comparando con
 * el segundo vecino, y utilizando la libreria FLANN mas segundo vecino.
 */
vector<DMatch> match_descriptors(Mat descriptors1, Mat descriptors2, int mode) {
	vector<DMatch> matches, good_matches;
	vector<vector<DMatch>> matches_vecinos;
	Ptr<BFMatcher> matcherBF;
	double distancia_minima = INFINITY;
	int num_matches;

	switch(mode){
	case USE_BRUTE:
		matcherBF = BFMatcher::create(NORM_HAMMING,true);
		matcherBF->match(descriptors1, descriptors2, matches);

		for (int i = 0; i <matches.size(); i++)
		{
			if (matches[i].distance < distancia_minima)
			{
				distancia_minima = matches[i].distance;
			}
		}
		num_matches = matches.size();
		for (int i = 0; i < num_matches; i++)
		{
			if (matches[i].distance <= max(3 * distancia_minima, 35.0))
			{
				good_matches.push_back(matches[i]);
			}
		}
		matches = good_matches;
		break;
	case USE_VECINO:
		matcherBF = BFMatcher::create(NORM_HAMMING, false);
		matcherBF->knnMatch(descriptors1, descriptors2, matches_vecinos,2);

		num_matches = matches_vecinos.size();
		for (int i = 0; i < num_matches; i++)
		{
			if ((matches_vecinos[i][0].distance <= 0.5*matches_vecinos[i][1].distance))
			{
				good_matches.push_back(matches_vecinos[i][0]);
			}
		}
		matches = good_matches;
		break;
	case USE_FLANN:
		/*if (descriptors1.type() != CV_32F) {
			descriptors1.convertTo(descriptors1, CV_32F);
		}
		if (descriptors2.type() != CV_32F) {
			descriptors2.convertTo(descriptors2, CV_32F);
		}*/

		Ptr<cv::DescriptorMatcher> matcher = makePtr<cv::FlannBasedMatcher>(makePtr<cv::flann::LshIndexParams>(12, 20, 2));
		
		matcher->knnMatch(descriptors1, descriptors2, matches_vecinos,2);

		num_matches = matches_vecinos.size();
		for (int i = 0; i < num_matches; i++)
		{
			if (((matches_vecinos[i].size() > 0)) && (matches_vecinos[i][0].distance <= 0.5*matches_vecinos[i][1].distance) && (matches_vecinos[i][0].distance <= 35))
			{
				good_matches.push_back(matches_vecinos[i][0]);
			}
		}
		matches = good_matches;
		break;
	}

	return matches;
}

/**
 * Elimina los bordes negros de una imagen.
 */
Mat cropBroders(Mat src) {
	Mat bt;
	cvtColor(src, bt, COLOR_BGR2GRAY);
	threshold(bt, bt, 1, 255, THRESH_BINARY);

	int xmin = bt.cols;
	int ymin = bt.rows;
	int xmax = 0;
	int ymax = 0;

	double pixel;
	for (int x = 10; x < bt.cols; x++) {
		for (int y = 10; y < bt.rows; y++) {

			pixel = bt.at<unsigned char>(y, x);

			if (pixel != 0)
			{
				if (x < xmin)
				{
					xmin = x;
				}

				if (y < ymin)
				{
					ymin = y;
				}

				if (x > xmax)
				{
					xmax = x;
				}

				if (y > ymax)
				{
					ymax = y;
				}
			}

		}
	}

	Rect roi = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
	Mat resultado(src, roi);
	return resultado;

}

/**
 * Genera el modelo de ransac siguiendo DTL
 */
Mat generarModelo(int * usados, vector<Point2f> puntos1, vector<Point2f> puntos2) {

}
/**
 * Calcula el ransac.
 */
Mat hom_ransac(vector<Point2f> puntos2, vector<Point2f> puntos1, Mat mask, int limite = 50) {
	Mat homography;
	int usados[TAM_MODELO], tam = puntos2.size(), numInlines, maxInlines = 0;

	for (int i = 0; i < limite; i++) {
		for (int z = 0; z < TAM_MODELO; z++) {
			usados[z] = rand() % tam;
		}
		Mat modelo = generarModelo(usados, puntos2, puntos1);

		for (int z = 0; z < puntos2.size(); z++) {
			int x = 1;
		}
	}

	return homography;
}
/**
 * Junta la imagen de la homografia la imagen original, siempre y cuando la union entre ambas
 * sea en la parte derecha de la imagen original.
 */
Mat juntarImagenes_derecha(Mat referencia_color, Mat nueva_color, Mat referencia, Mat nueva, int ultima_columna, int filtro = 1) {
	nueva.resize(referencia.rows);
	nueva_color.resize(referencia.rows);

	double alpha = 0.5;
	Mat imagenFinal(referencia.rows, referencia.cols + nueva.cols, DataType<unsigned char>::type);
	Mat imagenFinal_color(referencia.rows, referencia.cols + nueva.cols, CV_8UC3);
	int posicion = -1;

	for (int i = 1; i < nueva.rows; ++i) {
		for (int j = 1; j < nueva.cols; ++j) {
			if (filtro == 1) {
				if ((i < referencia.rows) && (j < referencia.cols) && (nueva.at<unsigned char>(i, j) != 0) && (referencia.at<unsigned char>(i, j) != 0)) {
					if (posicion < 0) {
						double espacio = ultima_columna - j;
						alpha = 1 / espacio;
						posicion = j;
					}
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j)*min(1,(alpha*abs(j - posicion))) + referencia_color.at<Vec3b>(i, j)*max(0,(1 - (alpha*abs(j - posicion))));
					
				}
				else if ((i < referencia.rows) && (j < referencia.cols) && (referencia.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else  {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
			else if (filtro == 0) {
				if ((i < referencia.rows) && (j < referencia.cols) && (referencia.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
		}
	}

	cv::imshow("Panoramix", cropBroders(imagenFinal_color));

	cv::waitKey(13);

	return cropBroders(imagenFinal_color);
}

/**
* Junta la imagen de la homografia la imagen original, siempre y cuando la union entre ambas
* sea abajo de la imagen original.
*/
Mat juntarImagenes_abajo(Mat referencia_color, Mat nueva_color, Mat referencia, Mat nueva, int ultima_fila, int filtro = 0) {
	double alpha = 0.5;
	Mat imagenFinal(nueva.rows + referencia.cols, nueva.cols + referencia.cols, DataType<unsigned char>::type);
	Mat imagenFinal_color(nueva.rows + referencia.rows, nueva.cols + referencia.cols, CV_8UC3);
	
	int posicion = -1;

	for (int i = 1; i < imagenFinal.rows; ++i) {
		for (int j = 1; j < imagenFinal.cols; ++j) {
			if (filtro == 1) {
				if ((i < nueva.rows) && (j < nueva.cols) && (nueva.at<unsigned char>(i, j) != 0) && (referencia.at<unsigned char>(i, j) != 0)) {
					if (posicion < 0) {
						double espacio = abs(ultima_fila - i);
						alpha = 1 / espacio;
						posicion = i;
					}
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j)*min(1, (alpha*abs(i - posicion))) + referencia_color.at<Vec3b>(i, j)*max(0, (1 - (alpha*abs(i - posicion))));

				}
				else if ((i < referencia.rows) && (j < referencia.cols) && (referencia.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else if ((i < nueva.rows) && (j < nueva.cols)){
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
			else if (filtro == 0) {
				if ((i < referencia.rows) && (j < referencia.cols) && (referencia.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else if ((i < nueva.rows) && (j < nueva.cols)) {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
		}
	}

	imagenFinal_color = cropBroders(imagenFinal_color);
	cv::imshow("Panoramix", imagenFinal_color);

	cv::waitKey(13);

	return imagenFinal_color;
}

/**
 * Junta la imagen de la homografia la imagen original, siempre y cuando la union entre ambas
 * sea en la parte izquierda de la imagen original.
 */
Mat juntarImagenes_izquierda(Mat referencia_color, Mat nueva_color, Mat referencia, Mat nueva, int ultima_columna, int filtro = 1) {
	nueva.resize(referencia.rows);
	nueva_color.resize(referencia.rows);

	Mat imagenFinal_color(referencia.rows, referencia.cols, CV_8UC3);
	Mat imagenFinal(referencia.rows, referencia.cols, CV_8U);

	int posicion = -1;
	double espacio, alpha = 0.5;
	for (int i = 1; i < imagenFinal_color.rows; ++i) {
		for (int j = 1; j < imagenFinal_color.cols; ++j) {
			if (filtro == 1) {
				//Miramos si estamos en zona de interseccion
				if ((nueva.at<unsigned char>(i , j) != 0) && (referencia.at<unsigned char>(i , j)!=0)) {
					if (posicion == -1) {
						espacio = abs(ultima_columna - j);
						alpha = 1 / espacio;
						posicion = j;
						cout << "Alpha: " << alpha << " Ultima Columna " << ultima_columna << " Posicion " << posicion << "Correcto: "<< abs(ultima_columna-posicion) * alpha<< endl;
					}
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i , j) *min(1,(alpha*abs(j - posicion))) + nueva_color.at<Vec3b>(i, j)*max(0,(1 - (alpha*abs(j - posicion))));
				}
				//Si no cogemos de la referencia
				else if (referencia.at<unsigned char>(i, j) != 0) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				//Si no, estamos en el sitio a anyadir (comprobamos rangos de matriz)
				else if ((i < nueva.rows) && (j < nueva.cols) && (nueva.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
			if (filtro == 0) {
				if ((nueva.at<unsigned char>(i, j) != 0) && (referencia.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				//Si no cogemos de la referencia
				else if (referencia.at<unsigned char>(i, j) != 0) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				//Si no, estamos en el sitio a anyadir (comprobamos rangos de matriz)
				else if ((i < nueva.rows) && (j < nueva.cols) && (nueva.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
		}
	}
	
	cv::imshow("Panoramix", cropBroders(imagenFinal_color));

	cv::waitKey(13);

	return cropBroders(imagenFinal_color);
}

/**
* Junta la imagen de la homografia la imagen original, siempre y cuando la union entre ambas
* sea en la parte arriba de la imagen original.
*/
Mat juntarImagenes_arriba(Mat referencia_color, Mat nueva_color, Mat referencia, Mat nueva, int ultima_fila, int filtro = 1) {
	Mat imagenFinal_color(referencia.rows, referencia.cols, CV_8UC3);
	Mat imagenFinal(referencia.rows, referencia.cols, CV_8U);

	int posicion = -1;
	double espacio, alpha = 0.5;
	for (int i = 1; i < referencia.rows; ++i) {
		for (int j = 1; j < referencia.cols - 1; ++j) {
			if (filtro == 1) {
				//Miramos si estamos en zona de interseccion
				if ((i < nueva.rows) && (j < nueva.cols) && (nueva.at<unsigned char>(i, j) != 0) && (referencia.at<unsigned char>(i, j) != 0)) {
					if (posicion == -1) {
						espacio = abs(ultima_fila - i);
						alpha = 1 / espacio;
						posicion = i;
						cout << "Alpha: " << alpha << " Ultima Columna " << ultima_fila << " Posicion " << posicion << " Correcto: " << abs(ultima_fila - posicion) * alpha << endl;
					}
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j) *min(1, (alpha*abs(i - posicion))) + referencia_color.at<Vec3b>(i, j)*max(0, (1 - (alpha*abs(i - posicion))));
				}
				//Si no cogemos de la referencia
				else if (referencia.at<unsigned char>(i, j) != 0) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				//Si no, estamos en el sitio a anyadir (comprobamos rangos de matriz)
				else if ((i < nueva.rows) && (j < nueva.cols) && (nueva.at<unsigned char>(i, j) != 0)) {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
			if (filtro == 0) {
				if (referencia.at<unsigned char>(i, j) != 0) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else if ((i < nueva.rows) && (j < nueva.cols)){
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
		}
	}

	cv::imshow("Panoramix", cropBroders(imagenFinal_color));

	cv::waitKey(13);

	return cropBroders(imagenFinal_color);
}
/**
 * Dadas dos imagenes en blanco y negro y en color, crea un panorama con ambas si encuentra los suficientes
 * puntos de interes.
 */
Mat match_images(Mat imagen1, Mat imagen1_color, Mat imagen2, Mat imagen2_color, bool full = true, int min_match = 10, int modo =  USE_BRISK, int filtro = 1) {
	vector<KeyPoint> k1, k2;
	vector<Point2f> puntos1, puntos2;
	vector<DMatch> matches, inliers;
	Mat mask, homografia, descriptores1, descriptores2, composicion, composicion_color, inliers_figure;
	k1 = compute_keypoints(imagen1, modo);
	k2 = compute_keypoints(imagen2, modo);
	descriptores1 = extraer_descriptores(imagen1, k1, modo);
	descriptores2 = extraer_descriptores(imagen2, k2, modo);

	matches = match_descriptors(descriptores1, descriptores2, USE_FLANN);

	if (matches.size() > min_match) {
		int num_matches = matches.size();
		for (int i = 0; i < num_matches; i++) {
			puntos1.push_back(k1[matches[i].queryIdx].pt);
			puntos2.push_back(k2[matches[i].trainIdx].pt);
		}

		homografia = findHomography(puntos2, puntos1, mask, CV_RANSAC);
		for (int i = 0; i < num_matches; i++) {
			if (mask.at<unsigned char>(i)) {
				inliers.push_back(matches[i]);
			}
		}

		if (full) {
			namedWindow("inliers", 1);
			drawMatches(imagen1, k1, imagen2, k2, matches, inliers_figure);
			imshow("inliers primeros", inliers_figure);
		}

		warpPerspective(imagen2, composicion, homografia, Size(imagen1.cols+imagen2.cols, imagen1.rows));
		warpPerspective(imagen2_color, composicion_color, homografia, Size(imagen1.cols + imagen2.cols, imagen1.rows));

		if (full) {
			namedWindow("Homografia primer", 2);
			imshow("Homografia primer", composicion);
		}
		Mat traspose;
		int suma_ultima_columna = 0;
		vector<int> v_sumas, r_sumas;
		
		reduce(composicion, v_sumas, 0, CV_REDUCE_SUM, -1);
		transpose(composicion, traspose);
		reduce(traspose, r_sumas, 0, CV_REDUCE_SUM, -1);
		int tam = v_sumas.size(), tam_r = r_sumas.size(), ultima_columna, ultima_fila;
		for (int z = 1; z < tam - 1; z++) {
			if ((v_sumas[z] == 0) && (v_sumas[z + 1] == 0) && (v_sumas[z - 1] != 0)) {
				ultima_columna = z - 1;
				break;
			}
		}
		for (int z = 1; z < tam - 1; z++) {
			if ((r_sumas[z] == 0) && (r_sumas[z + 1] == 0) && (r_sumas[z - 1] != 0)) {
				ultima_fila = z - 1;
				break;
			}
		}

		if ((v_sumas[0] == 0) && (ultima_columna > (tam -1)/2) && (r_sumas[10] != 0) && (r_sumas[r_sumas.size() - 10] != 0)) {
			cout << "Empalmamos por la derecha." << endl;
			return juntarImagenes_derecha(imagen1_color, composicion_color, imagen1, composicion, ultima_columna, filtro);
		}
		else {
			//Reajustamos a un marco mas grande (tam actual + tam actual*coef_aum)
			float coef_aum = 2;
			int aumentoC1 = imagen1.cols * coef_aum, aumentoR1 = imagen1.rows * coef_aum;
			int aumentoC2 = imagen2.cols * coef_aum, aumentoR2 = imagen2.rows * coef_aum;

			Mat n_m2(imagen2.rows + aumentoR2, imagen2.cols + aumentoC2, DataType<unsigned char>::type, Scalar(0));
			Mat n_m2_color(imagen2.rows + aumentoR2, imagen2.cols + aumentoC2, CV_8UC3, Vec3b(0, 0, 0));
			Mat n_m1(imagen1.rows + aumentoR1, imagen1.cols + aumentoC2, DataType<unsigned char>::type, Scalar(0));
			Mat n_m1_color(imagen1.rows + aumentoR1, imagen1.cols + aumentoC1, CV_8UC3, Vec3b(0, 0, 0));


			for (int i = 0; i < imagen2.rows; i++) {
				for (int j = 0; j < imagen2.cols; j++) {
					n_m2_color.at<Vec3b>(i + aumentoR2 / 2, j + aumentoC2 / 2) = imagen2_color.at<Vec3b>(i, j);
					n_m2.at<unsigned char>(i + aumentoR2 / 2, j + aumentoC2 / 2) = imagen2.at<unsigned char>(i, j);

				}
			}
			for (int i = 0; i < imagen1.rows; i++) {
				for (int j = 0; j < imagen1.cols; j++) {
					n_m1_color.at<Vec3b>(i + aumentoR1 / 2, j + aumentoC1 / 2) = imagen1_color.at<Vec3b>(i, j);
					n_m1.at<unsigned char>(i + aumentoR1 / 2, j + aumentoC1 / 2) = imagen1.at<unsigned char>(i, j);
				}
			}

			//Como hemos movido las imagenes respecto del marco, hay que recalcular la homografia.
			for (int i = 0; i < k1.size(); i++) {
				k1[i].pt.y += aumentoR1 / 2;
				k1[i].pt.x += aumentoC1 / 2;
			}
			for (int i = 0; i < k2.size(); i++) {
				k2[i].pt.y += aumentoR2 / 2;
				k2[i].pt.x += aumentoC2 / 2;
			}

			if (full) {
				drawMatches(n_m1, k1, n_m2, k2, inliers, inliers_figure);
				imshow("inliers_new", cropBroders(inliers_figure));
			}

			puntos1.clear();
			puntos2.clear();

			int num_matches = matches.size();
			for (int i = 0; i < num_matches; i++) {
				puntos1.push_back(k1[matches[i].queryIdx].pt);
				puntos2.push_back(k2[matches[i].trainIdx].pt);
			}

			homografia = findHomography(puntos2, puntos1, mask, CV_RANSAC);

			warpPerspective(n_m2, composicion, homografia, Size(n_m2.cols, n_m2.rows));
			warpPerspective(n_m2_color, composicion_color, homografia, Size(n_m2.cols, n_m2.rows));

			if (full) {
				namedWindow("Homografia", 2);
				imshow("Homografia", composicion);
			}

			if ((v_sumas[0] != 0) && (ultima_columna < tam - 1) && (r_sumas[10] != 0) && (r_sumas[r_sumas.size() - 10] != 0)) {
				reduce(composicion, v_sumas, 0, CV_REDUCE_SUM, -1);
				tam = v_sumas.size();
				for (int z = 1; z < tam - 1; z++) {
					if ((v_sumas[z] == 0) && (v_sumas[z + 1] == 0) && (v_sumas[z - 1] != 0)) {
						ultima_columna = z - 1;
						break;
					}
				}
				cout << "Empalmamos por la izquierda." << endl;
				return juntarImagenes_izquierda(n_m1_color, composicion_color, n_m1, composicion, ultima_columna, filtro);
			}
			else if ((r_sumas[0] != 0) && (r_sumas[r_sumas.size() - 5] == 0)) {
				transpose(composicion, traspose);
				reduce(traspose, r_sumas, 0, CV_REDUCE_SUM, -1);
				tam = r_sumas.size();
				for (int z = 1; z < tam - 1; z++) {
					if ((r_sumas[z] == 0) && (r_sumas[z + 1] == 0) && (r_sumas[z - 1] != 0)) {
						ultima_fila = z - 1;
						break;
					}
				}
				cout << "Empalmamos por arriba. " << ultima_fila << endl;
				return juntarImagenes_arriba(n_m1_color, composicion_color, n_m1, composicion, ultima_fila, filtro);
			}
			else {
				transpose(composicion, traspose);
				reduce(traspose, r_sumas, 0, CV_REDUCE_SUM, -1);
				tam = r_sumas.size();
				for (int z = 1; z < tam - 1; z++) {
					if ((r_sumas[z] == 0) && (r_sumas[z + 1] == 0) && (r_sumas[z - 1] != 0)) {
						ultima_fila = z - 1;
						break;
					}
				}
				cout << "Empalmamos por abajo." << endl;
				return juntarImagenes_abajo(n_m1_color, composicion_color, n_m1, composicion, ultima_fila,filtro);
			}
		}
		
	}
	else {
		cout << "No hay suficientes puntos para hacer el panorama!" << endl;
		return imagen1_color;
	}

}

/**
 * Permite capturar x imagenes con la camara para realizar un panorama con ellas.
 */
void matching_camera_boton(int numero_imagenes = 2) {
	VideoCapture cap;
	Mat imagen1_color, imagen1, imagen2_color, imagen2;
	int tomadas = 0;
	bool terminar = false;
	namedWindow("Panoramix", 7);
	namedWindow("Camara", 3);

	if (!cap.open(0))
		return ;
	while (tomadas < numero_imagenes)
	{
		cout << "Preparado para capturar..." << endl;
		while (!terminar) {
			if (tomadas == 0) {
				cap >> imagen1_color;
				imshow("Camara", imagen1_color);
			}
			else {
				cap >> imagen2_color;
				imshow("Camara", imagen2_color);
			}
			terminar = cv::waitKey(10) == 13;
		}
		terminar = false;
		cout << "Imagen capturada!..." << endl;
		if (tomadas == 0) {
			cvtColor(imagen1_color, imagen1, CV_BGR2GRAY);
			cv::imshow("Panoramix", imagen1_color);
		}
		else {
			cvtColor(imagen2_color, imagen2, CV_BGR2GRAY);
			imagen1_color = match_images(imagen1, imagen1_color, imagen2, imagen2_color);
			cvtColor(imagen1_color, imagen1, CV_BGR2GRAY);
		}
		tomadas++;
	}
	
}

/**
 * Captura imagenes con la camara y construye automaticamente un panorama con ellas si es posible.
 * Se puede especificar un tiempo de ciclo para escoger cada cuanto se capturan las imagenes.
 */
void matching_camera_automatico(int ciclo = 1000) {
	VideoCapture cap;
	Mat imagen1_color, imagen1, imagen2_color, imagen2;
	int tomadas = 0;
	bool terminar = false;
	

	if (!cap.open(0))
		return;
	waitKey();
	while (true) {
		if (tomadas == 0) {
			cap >> imagen1_color;
			imshow("Camara", imagen1_color);
		}
		else {
			cap >> imagen2_color;
			imshow("Camara", imagen2_color);
		}
			
		if (tomadas == 0) {
			cvtColor(imagen1_color, imagen1, CV_BGR2GRAY);
			cv::imshow("Panoramix", imagen1_color);
		}
		else {
			cvtColor(imagen2_color, imagen2, CV_BGR2GRAY);
			imagen1_color = match_images(imagen2, imagen2_color, imagen1, imagen1_color);
			cvtColor(imagen1_color, imagen1, CV_BGR2GRAY);
			cv::imshow("Panoramix", imagen1_color);
		}
		tomadas++;
		Sleep(ciclo);

		cout << "pasa ciclo" << endl;
	}

}

/**
 * Calcula el panorama de dos imagenes que estan en disco.
 */
void matching_disco(char ** argv, bool full = false) {
	Mat imagen1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE), imagen1_color = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	Mat imagen2 = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE), imagen2_color = imread(argv[3], CV_LOAD_IMAGE_COLOR);

	match_images(imagen1, imagen1_color, imagen2, imagen2_color);
}
/**
 * Calcula un panorama de imagenes tomadas segun el parametro dado:
 * -d Las carga de disco (se requieren las dos rutas de las imagenes)
 * -b Las carga de la camara pulsando enter.
 * -v Las carga de la camara en tiempo real.
 */
void main(int argc, char ** argv) {
	if (strcmp(argv[1],"-d") == 0) {
		matching_disco(argv);
	}
	else if (strcmp(argv[1], "-b") == 0) {
		matching_camera_boton(10);
	}
	else if (strcmp(argv[1], "-v") == 0) {
		matching_camera_automatico();
	}
	

	cv::waitKey(0);
}