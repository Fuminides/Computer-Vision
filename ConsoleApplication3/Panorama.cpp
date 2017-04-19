#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define USE_BRISK 0
#define USE_ORB 1
#define USE_FAST 2

#define USE_BRUTE 3
#define USE_VECINO 4
#define USE_FLANN 5

vector<KeyPoint> compute_keypoints(Mat imagen, int mode) {
	Ptr<FeatureDetector> detector;
	switch (mode) {
	case USE_FAST:
		detector = FastFeatureDetector::create(40);
		break;
	case USE_BRISK:
		detector = BRISK::create();
		break;
	}

	vector<KeyPoint> keypoints;
	detector->detect(imagen, keypoints);

	return keypoints;
}

Mat extraer_descriptores(Mat imagen, vector<KeyPoint> keypoints, int mode) {
	Ptr<DescriptorExtractor> descriptorExtractor;
	switch (mode) {
	case USE_BRISK:
		descriptorExtractor = BRISK::create();
		break;
	case USE_ORB:
		descriptorExtractor = ORB::create();
		break;
	}
	Mat descriptors;

	descriptorExtractor->compute(imagen, keypoints, descriptors);

	return descriptors;
}

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

		for (int i = 0; i < descriptors1.rows; i++)
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
			if (matches_vecinos[i][0].distance <= 0.5*matches_vecinos[i][1].distance)
			{
				good_matches.push_back(matches_vecinos[i][0]);
			}
		}
		matches = good_matches;
		break;
	case(USE_FLANN):
		if (descriptors1.type() != CV_32F) {
			descriptors1.convertTo(descriptors1, CV_32F);
		}
		if (descriptors2.type() != CV_32F) {
			descriptors2.convertTo(descriptors2, CV_32F);
		}
		Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
		matcher->knnMatch(descriptors1, descriptors2, matches_vecinos,2);

		num_matches = matches_vecinos.size();
		for (int i = 0; i < num_matches; i++)
		{
			if (matches_vecinos[i][0].distance <= 0.5*matches_vecinos[i][1].distance)
			{
				good_matches.push_back(matches_vecinos[i][0]);
			}
		}
		matches = good_matches;
		break;
	}

	//for(int i = 0; i < matches.size(); i++) cout << matches[i].distance << " " << matches[i].imgIdx << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
	return matches;
}

Mat juntarImagenes_derecha(Mat referencia_color, Mat nueva_color, Mat referencia, Mat nueva, vector<KeyPoint> k2, Mat descriptores2, int ultima_columna, int modo = USE_BRISK, int filtro = 1) {
	
	
	if (filtro == 2) {
		Mat ilum_referencia, ilum_nueva;
		vector<Mat> channels_referencia, channels_nueva;
		Scalar media_referencia, media_nueva;
		double proporcion;

		cvtColor(referencia_color, ilum_referencia, CV_BGR2YCrCb);
		cvtColor(nueva_color, ilum_nueva, CV_BGR2YCrCb);
		split(ilum_referencia, channels_referencia);
		split(ilum_nueva, channels_nueva);
		media_referencia = mean(channels_referencia[0]);
		media_nueva = mean(channels_nueva[0]);
		proporcion = media_referencia[0] / media_nueva[0];
		cout << proporcion << endl;
		channels_nueva[0] = channels_nueva[0] * proporcion*0.80;
		merge(channels_nueva, ilum_nueva);
		cvtColor(ilum_nueva, nueva_color, CV_YCrCb2BGR);
	}

	nueva.resize(referencia.rows);
	nueva_color.resize(referencia.rows);
	double alpha = 0.5;
	Mat imagenFinal(referencia.rows, referencia.cols + nueva.cols, DataType<unsigned char>::type);
	Mat imagenFinal_color(referencia.rows, referencia.cols + nueva.cols, CV_8UC3);
	int posicion = -1;
	for (int i = 1; i < referencia.rows; ++i) {
		for (int j = 1; j < ultima_columna; ++j) {
			if (filtro == 1) {
				if ((nueva.at<unsigned char>(i, j) != 0) && (j < referencia.cols - 1)) {
					if (posicion < 0) {
						double espacio = referencia.cols - j;
						alpha = 1 / espacio;
						posicion = j;
					}
					imagenFinal.at<unsigned char>(i, j) = referencia.at<unsigned char>(i, j);
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j)*(alpha*abs(j - posicion)) + referencia_color.at<Vec3b>(i, j)*(1 - (alpha*abs(j - posicion)));
				}
				else if (j < referencia.cols - 1) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
			else if (filtro == 2) {
				if (j < referencia.cols - 1) {
					imagenFinal_color.at<Vec3b>(i, j) = referencia_color.at<Vec3b>(i, j);
				}
				else {
					imagenFinal_color.at<Vec3b>(i, j) = nueva_color.at<Vec3b>(i, j);
				}
			}
		}
	}

	imagenFinal_color = imagenFinal_color.colRange(0, ultima_columna);
	cv::namedWindow("Panoramix", 7);
	cv::imshow("Panoramix", imagenFinal_color);

	cv::waitKey(0);

	return imagenFinal;
}

void match_images(Mat imagen1, Mat imagen1_color, Mat imagen2, Mat imagen2_color, bool full = true) {
	vector<KeyPoint> k1, k2;
	vector<Point2f> puntos1, puntos2;
	vector<DMatch> matches, inliers;
	Mat mask, homografia, descriptores1, descriptores2, composicion, composicion_color, inliers_figure;

	k1 = compute_keypoints(imagen1, USE_BRISK);
	k2 = compute_keypoints(imagen2, USE_BRISK);
	descriptores1 = extraer_descriptores(imagen1, k1, USE_BRISK);
	descriptores2 = extraer_descriptores(imagen2, k2, USE_BRISK);

	matches = match_descriptors(descriptores1, descriptores2, USE_VECINO);

	if (full) {
		int num_matches = matches.size();
		for (int i = 0; i < num_matches; i++) {
			puntos1.push_back(k1[matches[i].queryIdx].pt);
			puntos2.push_back(k2[matches[i].trainIdx].pt);
		}

		homografia = findHomography(puntos1, puntos2, mask, CV_RANSAC);
		for (int i = 0; i < num_matches; i++) {
			if (mask.at<unsigned char>(i)) {
				inliers.push_back(matches[i]);
			}
		}
		namedWindow("inliers", 1);
		drawMatches(imagen1, k1, imagen2, k2, inliers, inliers_figure);
		imshow("inliers", inliers_figure);
		warpPerspective(imagen1, composicion, homografia, Size(imagen1.cols+imagen2.cols, imagen1.rows));
		warpPerspective(imagen1_color, composicion_color, homografia, Size(imagen1.cols + imagen2.cols, imagen1.rows));

		namedWindow("Homografia", 2);
		imshow("Homografia", composicion);
		int suma_ultima_columna = 0;
		vector<int> v_sumas;
		reduce(composicion, v_sumas, 0, CV_REDUCE_SUM, -1);
		int tam = v_sumas.size(), ultima_columna;
		for (int z = 1; z < tam-1; z++) {
			if ((v_sumas[z] == 0) && (v_sumas[z+1] == 0) && (v_sumas[z-1] != 0)) {
				ultima_columna = z - 1;
				break;
			}
		}
		juntarImagenes_derecha(imagen2_color, composicion_color, imagen2, composicion, k2, descriptores2, ultima_columna, USE_BRISK);

	}
	else {
		namedWindow("matches", 1);
		Mat img_matches;
		drawMatches(imagen1, k1, imagen2, k2, matches, img_matches);
		imshow("matches", img_matches);
		waitKey(0);
	}

}

void matching_disco(int argc, char ** argv, bool full = false) {
	Mat imagen1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE), imagen1_color = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat imagen2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE), imagen2_color = imread(argv[2], CV_LOAD_IMAGE_COLOR);

	match_images(imagen1, imagen1_color, imagen2, imagen2_color);
}
void main(int argc, char ** argv) {
	matching_disco(argc, argv, true);
}