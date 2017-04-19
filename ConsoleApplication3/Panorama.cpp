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


void matching_disco(int argc, char ** argv, bool full=false) {
	Mat imagen1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat imagen2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	vector<KeyPoint> k1, k2;
	vector<Point2f> puntos1, puntos2;
	vector<DMatch> matches, inliers;
	Mat mask, homografia, descriptores1, descriptores2, composicion, inliers_figure;

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
		
		warpPerspective(imagen1, composicion, homografia, imagen1.size());
		namedWindow("Homografia", 2);
		imshow("Homografia", composicion);

		//La 1 se convierte en la 2
		int y_stitch = 0, y_final=0;
		int inliers_size = inliers.size();
		k1 = compute_keypoints(composicion, USE_BRISK);
		descriptores1 = extraer_descriptores(composicion, k1, USE_BRISK);
		matches = match_descriptors(descriptores2, descriptores1, USE_VECINO);

		for (int i = 0; i < inliers_size; i++) {
			Point2f good_match = k2[matches[i].queryIdx].pt;
			cout << "PT: " << good_match << endl;
			if (good_match.y > y_final) {
				y_final = good_match.y;
				y_stitch = k2[inliers[i].trainIdx].pt.y;
				
			}
		}
		cout << "y_final" << y_final << " y empalme " << y_stitch << endl;
		/*namedWindow("matches finales", 6);
		Mat matches_finales;
		//drawMatches(imagen2, k1, composicion, k2, matches, matches_finales);
		imshow("matches finales", matches_finales);*/
		Mat imagenFinal(imagen1.rows,  y_final + imagen1.cols - y_stitch, DataType<unsigned char>::type);
		for (int i = 0; i < imagenFinal.rows; ++i) {
			for (int j = 0; j < imagenFinal.cols; ++j) {
				if (j > y_final) {
					imagenFinal.at<unsigned char>(i, j) = composicion.at<unsigned char>(i, j - y_final + y_stitch);
				}
				else {
					imagenFinal.at<unsigned char>(i,j) = imagen2.at<unsigned char>(i, j);
				}
			}
		}
		namedWindow("Panoramix", 4);
		imshow("Panoramix", imagenFinal);

		waitKey(0);

	}
	else {
		namedWindow("matches", 1);
		Mat img_matches;
		drawMatches(imagen1, k1, imagen2, k2, matches, img_matches);
		imshow("matches", img_matches);
		waitKey(0);
	}
	


}

void main(int argc, char ** argv) {
	matching_disco(argc, argv, true);
}