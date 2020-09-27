#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <FreeImage.h>
#include <yzLib/yz_lib.h>

yz::opengl::DemoWindowManager	manager;
yz::opengl::DemoWindow3D		win3d;

yz::opengl::Shader				shader;

//	this shader get vertex position in camera coordinate and normal in KinectFusion global coordinate
static const char vertex_shader[] =
"\
	#version 330 compatibility\n \
	#extension GL_EXT_gpu_shader4 : enable\n \
	//out vec4 point;\n \
	void main()\n \
	{\n \
		gl_Position = ftransform();\n \
		gl_FrontColor = gl_Color;\n \
		//gl_Position = gl_ModelViewMatrix * gl_Vertex;\n \
		//gl_Position.y = - gl_Position.y;\n \
		//point = gl_ModelViewMatrix * gl_Vertex;\n \
	}\
";

static const char fragment_shader[] =
"\
	#version 330 compatibility\n \
	#extension GL_EXT_gpu_shader4 : enable\n \
	//in vec4 point;\n \
	void main()\n \
	{\n \
		gl_FragColor = gl_Color;\n \
		//gl_FragColor = vec4(1,1,0,1);\n \
		//gl_FragColor[0] = - point.z;\n \
	}\
";


void print3d() {
	glColor3f(1, 0, 0);
	yz::opengl::printInfo(0, 0, "fov_y=%f", win3d.fovy);
	yz::opengl::printInfo(0, 20, "spin_x=%f", win3d.spin_x);
	yz::opengl::printInfo(0, 40, "spin_y=%f", win3d.spin_y);
}

void draw3d() {
	yz::opengl::drawXYZAxis();

	shader.UseMe();

	glColor3f(1, 1, 0);
	yz::opengl::drawPointAsCube(yz::Vec3f(0, 0, 0), 0.5f);

	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);
	yz::opengl::drawPlaneWire(yz::Vec3f(0, 0, 0), yz::Vec3f(0, 0, 1), 5.0);

	shader.UseDefault();
}


int main() {
	win3d.SetDraw(draw3d);
	win3d.SetDrawAppend(print3d);
	win3d.CreateGLUTWindow();
	glewInit();

	shader.Setup(vertex_shader, fragment_shader);

	manager.EnterMainLoop();
}