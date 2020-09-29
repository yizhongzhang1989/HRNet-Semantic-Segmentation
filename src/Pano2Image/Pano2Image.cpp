#include "data_config.h"

#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <FreeImage.h>
#include <yzLib/yz_lib.h>

#include "ArgsParser.h"
#include "Panorama.h"

yz::opengl::DemoWindowManager	manager;
yz::opengl::DemoWindow3D		win3d;
yz::opengl::FBO					fbo;

PanoSphere	pano_sphere;
Ground		ground;
std::vector<std::pair<std::string, std::string>>	panorama_image_label;

int			sample_index = 0;
int			panorama_index = 0;
int			ground_tex_index = 0;
float		ground_tex_scale = 1.0f;
int			samples_per_panorama = 12;
int			image_width = 1024;
int			image_height = 768;
float		fov_min = 90.0f;
float		fov_max = 107.0f;
bool		rand_tilt = true;
bool		rand_ground = true;

bool		draw_label_flag = false;
const char* output_dir = "E:/StoreSemanticLabelingData";

void draw3d() {
	glDisable(GL_LIGHTING);

	if (draw_label_flag) {
		glColor4f(1, 1, 1, 1);
		pano_sphere.DrawLabel();
	}
	else {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glColor4f(1, 1, 1, 1);
		ground.Draw(ground_tex_index, ground_tex_scale);

		glColor4f(1, 1, 1, 1);
		pano_sphere.DrawColor(ground_tex_index >= 0);
	}
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
	}

	glutPostRedisplay();
}

void idle() {
	if (panorama_index >= panorama_image_label.size())
		return;

	//	calculate render parameters
	float fovy = yz::randFloatingPointNumber(fov_min, fov_max);
	win3d.fovy = fovy;

	float spinx = yz::randFloatingPointNumber(0.0f, 360.0f);
	win3d.spin_x = spinx;

	float max_tilt = -0.529f * fovy + 64.603f;
	float spiny = yz::randFloatingPointNumber(-max_tilt, max_tilt);
	win3d.spin_y = rand_tilt ? spiny : 0.0f;

	ground_tex_index = rand_ground ? yz::randNumber(0, int(ground.ground_textures.size())) - 1 : -1;

	ground_tex_scale = yz::randFloatingPointNumber(0.5f, 1.5f);

	//	render label
	draw_label_flag = true;
	fbo.BeginRender();

	yz::opengl::pushAllAttributesAndMatrices();
	win3d.DefaultReshapeFunc(win3d.win_width, win3d.win_height);
	win3d.auto_swap_buffers = 0;
	win3d.DefaultDisplayFunc();
	win3d.auto_swap_buffers = 1;
	yz::opengl::popAllAttributesAndMatrices();
	
	std::vector<yz::uchar3> label;
	label.resize(win3d.win_width * win3d.win_height);
	glReadPixels(0, 0, win3d.win_width, win3d.win_height, GL_RGB, GL_UNSIGNED_BYTE, &label[0].x);
	fbo.EndRender();

	//	render image
	draw_label_flag = false;
	fbo.BeginRender();

	yz::opengl::pushAllAttributesAndMatrices();
	win3d.DefaultReshapeFunc(win3d.win_width, win3d.win_height);
	win3d.auto_swap_buffers = 0;
	win3d.DefaultDisplayFunc();
	win3d.auto_swap_buffers = 1;
	yz::opengl::popAllAttributesAndMatrices();

	std::vector<yz::uchar3> image;
	image.resize(win3d.win_width * win3d.win_height);
	glReadPixels(0, 0, win3d.win_width, win3d.win_height, GL_RGB, GL_UNSIGNED_BYTE, &image[0].x);
	fbo.EndRender();

	//	dump data
	for (int j = 0; j < win3d.win_height / 2; j++) {
		for (int i = 0; i < win3d.win_width; i++) {
			yz::mySwap(label[j * win3d.win_width + i], label[(win3d.win_height - 1 - j) * win3d.win_width + i]);
			yz::mySwap(image[j * win3d.win_width + i], image[(win3d.win_height - 1 - j) * win3d.win_width + i]);
		}
	}

	char label_filename[256], image_filename[256];
	sprintf(label_filename, "%s/%d.png", output_dir, sample_index);
	sprintf(image_filename, "%s/%d.jpg", output_dir, sample_index);
	yz::image::writeImageToFile(label_filename, &label[0].x, win3d.win_width, win3d.win_height, 24);
	yz::image::writeImageToFile(image_filename, &image[0].x, win3d.win_width, win3d.win_height, 24);

	//	prepare next frame
	sample_index++;
	int next_panorama_index = sample_index / samples_per_panorama;
	if (next_panorama_index != panorama_index && next_panorama_index < panorama_image_label.size()) {
		panorama_index = next_panorama_index;
		pano_sphere.ReadTexture(panorama_image_label[panorama_index].first.c_str(), panorama_image_label[panorama_index].second.c_str());
		pano_sphere.color_tex.LoadPtrToTexture();
		pano_sphere.label_tex.LoadPtrToTexture();

		std::cout << "next panorama index: " << panorama_index 
			<< "  current dumped: " << sample_index
			<< std::endl;
	}

	glutPostRedisplay();
}

int main(int argc, const char** argv) {
	//	initialize data
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

	//	read arguments
	ArgsParser	parser(argc, argv);
	parser.TryGetArgment("samples_per_panorama", samples_per_panorama);
	parser.TryGetArgment("image_width", image_width);
	parser.TryGetArgment("image_height", image_height);
	parser.TryGetArgment("fov_min", fov_min);
	parser.TryGetArgment("fov_max", fov_max);
	parser.TryGetArgment("rand_tilt", rand_tilt);
	parser.TryGetArgment("rand_ground", rand_ground);

	//	create context
	win3d.eye_z = 0;
	win3d.fovy = 100.0f;
	win3d.motionFunc = motion;
	win3d.keyboardFunc = keyboard;
	win3d.specialFunc = special;
	win3d.SetWindowPositionSize(0, 0, image_width, image_height);
	win3d.SetDraw(draw3d);
	win3d.CreateGLUTWindow();
	glewInit();

	fbo.InitFBO(image_width, image_height);

	ground.CreateTextures();
	pano_sphere.CreateTexture();

	manager.AddIdleFunc(idle);
	manager.EnterMainLoop();
}
