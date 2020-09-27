#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <FreeImage.h>
#include <yzLib/yz_lib.h>

yz::opengl::DemoWindowManager	manager;
yz::opengl::DemoWindow3D		win3d;

yz::opengl::Shader				shader;


class PanoSphere : public yz::geometry::SingleDisplayTextureTriMesh<float> {
public:
	void ReadGeometry(const char* obj_filename) {
		ReadMeshFromFile(obj_filename);
	}

	void ReadTexture(const char* color_img_filename, const char* label_img_filename) {
		unsigned char *color_img_ptr, *label_img_ptr;
		int col_w, col_h, lab_w, lab_h;
		yz::image::readImageFromFile(color_img_filename, color_img_ptr, col_w, col_h, 24);
		yz::image::readImageFromFile(label_img_filename, label_img_ptr, lab_w, lab_h, 24);

		if (col_w != lab_w || col_h != lab_h) {
			std::cout << "color size don't match label size" << std::endl;
			return;
		}

		img_w = col_w;
		img_h = col_h;
		color_img.resize(img_w * img_h);
		label_img.resize(img_w * img_h);

		for (int j = 0; j < img_h; j++)
			for (int i = 0; i < img_w; i++) {
				int pid = j*img_w + i;
				int pid_flip = (img_h - 1 - j)*img_w + (img_w - 1 - i);
				label_img[pid].x = label_img_ptr[pid_flip * 3 + 0];
				label_img[pid].y = label_img_ptr[pid_flip * 3 + 1];
				label_img[pid].z = label_img_ptr[pid_flip * 3 + 2];

				color_img[pid].x = color_img_ptr[pid_flip * 3 + 0];
				color_img[pid].y = color_img_ptr[pid_flip * 3 + 1];
				color_img[pid].z = color_img_ptr[pid_flip * 3 + 2];

				if (label_img[pid].x == 0 && label_img[pid].y == 100 && label_img[pid].z == 100)
					color_img[pid].w = 128;
				else
					color_img[pid].w = 255;
			}
	}

	void CreateTexture() {
		color_tex.CreateTexture();
		color_tex.SetupTexturePtr(img_w, img_h, &color_img[0].x, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
		color_tex.LoadPtrToTexture();

		label_tex.CreateTexture();
		label_tex.SetupTexturePtr(img_w, img_h, &label_img[0].x, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
		label_tex.LoadPtrToTexture();
	}

	void DrawColor() {
		if (color_tex.tex_id)
			color_tex.Bind();

		if (!display_vertex.empty() && !display_vertex_normal.empty()) {
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, &display_vertex[0]);
			glEnableClientState(GL_NORMAL_ARRAY);
			glNormalPointer(GL_FLOAT, 0, &display_vertex_normal[0]);
		}

		if (!display_tex_coord.empty()) {
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glTexCoordPointer(2, GL_FLOAT, 0, &display_tex_coord[0]);
		}

		if (!display_face.empty()) {
			glColor3f(1, 1, 1);
			glDrawElements(GL_TRIANGLES, display_face.size() * 3, GL_UNSIGNED_INT, &display_face[0]);
		}

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

		if (color_tex.tex_id)
			color_tex.UnBind();
	}

	void DrawLabel() {
		if (label_tex.tex_id)
			label_tex.Bind();

		if (!display_vertex.empty() && !display_vertex_normal.empty()) {
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, &display_vertex[0]);
			glEnableClientState(GL_NORMAL_ARRAY);
			glNormalPointer(GL_FLOAT, 0, &display_vertex_normal[0]);
		}

		if (!display_tex_coord.empty()) {
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glTexCoordPointer(2, GL_FLOAT, 0, &display_tex_coord[0]);
		}

		if (!display_face.empty()) {
			glColor3f(1, 1, 1);
			glDrawElements(GL_TRIANGLES, display_face.size() * 3, GL_UNSIGNED_INT, &display_face[0]);
		}

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

		if (label_tex.tex_id)
			label_tex.UnBind();
	}

public:
	int img_w, img_h;
	std::vector<yz::uchar4>	color_img;
	std::vector<yz::uchar3> label_img;

	yz::opengl::Texture	color_tex, label_tex;
};

class Ground {
public:
	void ReadTextures(const char* dir) {
		std::string tex_list_filename = yz::utils::getFileNameCombineDirentory(dir, "tex_list.txt");
		std::ifstream tex_list_file(tex_list_filename.c_str());
		if (!tex_list_file.is_open()) {
			std::cout << "failed to open " << tex_list_filename << std::endl;
			return;
		}

		std::string img_name;
		float img_size;
		while (tex_list_file >> img_name >> img_size) {
			std::string img_filename = yz::utils::getFileNameCombineDirentory(dir, img_name.c_str());			

			unsigned char* img_ptr = NULL;;
			int w, h;
			if (yz::image::readImageFromFile(img_filename.c_str(), img_ptr, w, h, 24)) {
				ground_textures.resize(ground_textures.size() + 1);
				ground_textures.back().tex_size = img_size;
				ground_textures.back().tex_w = w;
				ground_textures.back().tex_h = h;
				ground_textures.back().tex_img.resize(w * h);
				memcpy(&ground_textures.back().tex_img[0].x, img_ptr, w * h * 3);
				delete[] img_ptr;
				img_ptr = NULL;
			}
		}
		tex_list_file.close();

		for (int i = 0; i < ground_textures.size(); i++) {
			std::cout << "tex " << i << ", ("
				<< ground_textures[i].tex_w << "*"
				<< ground_textures[i].tex_h << ")" << std::endl;
		}
	}

	void CreateTextures() {
		for (int i = 0; i < ground_textures.size(); i++) {
			ground_textures[i].tex.CreateTexture();
			ground_textures[i].tex.SetupTexturePtr(
				ground_textures[i].tex_w,
				ground_textures[i].tex_h,
				&ground_textures[i].tex_img[0].x,
				GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
			ground_textures[i].tex.LoadPtrToTexture();
		}
	}

	void Draw(int tex_index = 0, float height = 1.45f) {
		if (tex_index < 0 || tex_index >= ground_textures.size() || !ground_textures[tex_index].tex.tex_id)
			return;

		yz::Vec3f v[4] = {
			yz::Vec3f(-100, -height, -100),
			yz::Vec3f(-100, -height, 100),
			yz::Vec3f(100, -height, 100),
			yz::Vec3f(100, -height, -100)
		};

		ground_textures[tex_index].tex.Bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		glBegin(GL_QUADS);
		for (int i = 0; i < 4; i++) {
			glTexCoord2f(v[i].x / ground_textures[tex_index].tex_size, v[i].z / ground_textures[tex_index].tex_size);
			glVertex3f(v[i].x, v[i].y, v[i].z);
		}
		glEnd();

		ground_textures[tex_index].tex.UnBind();
	}

public:
	struct Tex {
		yz::opengl::Texture tex;

		float tex_size = 1.0f;
		int tex_w, tex_h;
		std::vector<yz::uchar3>	tex_img;		
	};

	std::vector<Tex> ground_textures;
};

PanoSphere	pano_sphere;
Ground		ground;
int			ground_tex_index = 0;

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
	pano_sphere.DrawColor();
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
	}

	glutPostRedisplay();
}

int main() {
	pano_sphere.ReadGeometry("../../../uv_sphere.obj");
	pano_sphere.ReadTexture("../../../location_02.jpg", "../../../location_02.png");
	ground.ReadTextures("../../../ground_tex");

	win3d.eye_z = 0;
	win3d.fovy = 100.0f;
	win3d.motionFunc = motion;
	win3d.specialFunc = special;
	win3d.SetDraw(draw3d);
	win3d.SetDrawAppend(print3d);
	win3d.CreateGLUTWindow();
	glewInit();

	ground.CreateTextures();
	pano_sphere.CreateTexture();

	manager.EnterMainLoop();
}