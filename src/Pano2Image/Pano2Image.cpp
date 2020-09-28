#include "data_config.h"

#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <FreeImage.h>
#include <yzLib/yz_lib.h>

#include "Panorama.h"

yz::opengl::DemoWindowManager	manager;
yz::opengl::DemoWindow3D		win3d(0, 0, 1024, 768);


PanoSphere	pano_sphere;
Ground		ground;
int			ground_tex_index = 0;
int			panorama_index = 0;
std::vector<std::pair<std::string, std::string>>	panorama_image_label;
int			draw_label_flag = 0;


void print3d() {
	glColor3f(1, 0, 0);
	yz::opengl::printInfo(0, 0, "fov_y=%f", win3d.fovy);
	yz::opengl::printInfo(0, 20, "spin_x=%f", win3d.spin_x);
	yz::opengl::printInfo(0, 40, "spin_y=%f", win3d.spin_y);
}

void draw3d() {
	glDisable(GL_LIGHTING);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glColor4f(1, 1, 1, 1);
	ground.Draw(ground_tex_index);

	glColor4f(1, 1, 1, 1);
	if (draw_label_flag)
		pano_sphere.DrawLabel();
	else
		pano_sphere.DrawColor(ground_tex_index >= 0);
}

void motion(int x, int y) {
	if (win3d.mouse_state[GLUT_LEFT_BUTTON] == GLUT_DOWN) {
		win3d.spin_x -= float(x - win3d.old_x);
		win3d.spin_y += float(y - win3d.old_y);
	}
	else if (win3d.mouse_state[GLUT_RIGHT_BUTTON] == GLUT_DOWN) {
		win3d.fovy += float(y - win3d.old_y) / 10;

		win3d.DefaultReshapeFunc(win3d.win_width, win3d.win_height);
	}

	win3d.old_x = x;
	win3d.old_y = y;
	glutPostRedisplay();
}

void special(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_UP:
		ground_tex_index++;
		break;
	case GLUT_KEY_DOWN:
		ground_tex_index--;
		break;
	case GLUT_KEY_LEFT:
		if (panorama_index > 0)
			panorama_index--;
		pano_sphere.ReadTexture(panorama_image_label[panorama_index].first.c_str(), panorama_image_label[panorama_index].second.c_str());
		pano_sphere.color_tex.LoadPtrToTexture();
		pano_sphere.label_tex.LoadPtrToTexture();
		break;
	case GLUT_KEY_RIGHT:
		if (panorama_index < panorama_image_label.size() - 1)
			panorama_index++;
		pano_sphere.ReadTexture(panorama_image_label[panorama_index].first.c_str(), panorama_image_label[panorama_index].second.c_str());
		pano_sphere.color_tex.LoadPtrToTexture();
		pano_sphere.label_tex.LoadPtrToTexture();
		break;
	}

	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		exit(0);
	case ' ':
		draw_label_flag = !draw_label_flag;
		break;
	}

	glutPostRedisplay();
}

int main() {
	std::string uv_sphere_filename = yz::utils::getFileNameCombineDirentory(ground_tex_dir, "../uv_sphere.obj");
	std::string pano_list_filename = yz::utils::getFileNameCombineDirentory(ground_tex_dir, "../bin/panorama_image_label_list.txt");
	std::ifstream	pano_list_file(pano_list_filename);
	if (pano_list_file.is_open()) {
		std::string image_filename, label_filename;
		while (pano_list_file >> image_filename >> label_filename) {
			panorama_image_label.push_back(std::pair<std::string, std::string>(image_filename, label_filename));
		}
	}
	if (panorama_image_label.empty()) {
		std::cout << "no panorama" << std::endl;
		return 0;
	}

	ground.ReadTextures(ground_tex_dir);
	pano_sphere.ReadGeometry(uv_sphere_filename.c_str());
	pano_sphere.ReadTexture(panorama_image_label[panorama_index].first.c_str(), panorama_image_label[panorama_index].second.c_str());

	win3d.eye_z = 0;
	win3d.fovy = 100.0f;
	win3d.motionFunc = motion;
	win3d.keyboardFunc = keyboard;
	win3d.specialFunc = special;
	win3d.SetDraw(draw3d);
	win3d.SetDrawAppend(print3d);
	win3d.CreateGLUTWindow();
	glewInit();

	ground.CreateTextures();
	pano_sphere.CreateTexture();

	manager.EnterMainLoop();
}
