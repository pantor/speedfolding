#include <string>
#include <vector>
#include <iostream>

#include <GL/osmesa.h>
#include <GL/glu.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "PhoXi.h"


namespace py = pybind11;


class NativePhoXiDriver {
    pho::api::PPhoXi m_scanner;
    pho::api::PhoXiFactory m_factory;

    std::string m_name;
    bool m_running;

    OSMesaContext ctx;
    void *buffer;
    int width, height;

public:
    NativePhoXiDriver(const std::string& name) : m_name(name), m_running(false) {
        ctx = OSMesaCreateContextExt(OSMESA_RGBA, 16, 0, 0, NULL);
        if (!ctx) {
            std::cout << "OSMesaCreateContext failed!" << std::endl;
            return;
        }
    }

    ~NativePhoXiDriver() {
        free(buffer);
    }

    bool start() {
        // m_factory.StartConsoleOutput("Admin-On");
        std::vector<pho::api::PhoXiDeviceInformation> dl = m_factory.GetDeviceList();
        if (dl.empty()) {
            std::cout << "[!] 0 devices found." << std::endl;
            return false;
        }

        bool connected = false;
        for (int i = 0; i < dl.size(); i++) {
            if (dl[i].HWIdentification == m_name) {
                connected = true;

                m_scanner = m_factory.Create(dl[i]);
                m_scanner = m_factory.CreateAndConnect(dl[i].HWIdentification, pho::api::PhoXiTimeout(10000));

                if (m_scanner && m_scanner->isConnected()) {
                    connected = true;
                }
            }
        }
    
        m_scanner = m_factory.CreateAndConnect(dl[0].HWIdentification, pho::api::PhoXiTimeout(10000));
        connected = m_scanner && m_scanner->isConnected();
        m_running = connected;
        return connected;
    }

    bool stop() {
        if (m_running) {
            m_scanner->Disconnect();
            m_running = false;
            return true;
        }
        return false;
    }

    void setOrthographicImageSize(int width, int height) {
        this->width = width;
        this->height = height;

        buffer = malloc(width * height * 4 * sizeof(GLfloat));
        if (!OSMesaMakeCurrent(ctx, buffer, GL_FLOAT, width, height)) {
            std::cout << "OSMesaMakeCurrent failed!" << std::endl;
            return;
        }
    }

    pho::api::PFrame get() {	
        if (!m_scanner || !m_scanner->isConnected()) {
            std::cout << "No scanner connected" << std::endl;
        }
        py::gil_scoped_release release;
        // m_scanner->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;

        bool success = m_scanner->isAcquiring();
        // std::cout << success;
        if (!m_scanner->isAcquiring()) {
            //using the below mode seems to crash the camera -justin kerr
            // m_scanner->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
            m_scanner->StartAcquisition();
        }

        if (!m_scanner->isAcquiring()) {
            std::cout << "Your device could not start acquisition!" << std::endl;
            return nullptr;
        }
        m_scanner->TriggerFrame();
        return m_scanner->GetFrame(pho::api::PhoXiTimeout(10000));
    }

    py::list read() {
        py::list data;
        if (!m_running) {
            return data;
        }

        pho::api::PFrame frame = get();

        if (!frame) {
            return data;
        }

        // Extract texture (grayscale image)
        size_t texture_w = frame->Texture.Size.Width;
        size_t texture_h = frame->Texture.Size.Height;
        float *texture = new float[texture_w * texture_h];
        frame->Texture.ConvertTo2DArray(texture, texture_h, texture_w);
        py::capsule free_texture(texture, [](void *f) {
            float *r = reinterpret_cast<float *>(f);
            delete[] r;
        });
        data.append(py::array_t<float>(
            {texture_h, texture_w},
            {texture_w*sizeof(float), sizeof(float)},
            texture,
            free_texture
        ));

        // Extract depth map
        size_t depth_w = frame->DepthMap.Size.Width;
        size_t depth_h = frame->DepthMap.Size.Height;
        float *depth = new float[depth_w * depth_h];
        frame->DepthMap.ConvertTo2DArray(depth, depth_h, depth_w);
        py::capsule free_depth(depth, [](void *f) {
            float *r = reinterpret_cast<float *>(f);
            delete[] r;
        });
        data.append(py::array_t<float>(
            {depth_h, depth_w},
            {depth_w*sizeof(float), sizeof(float)},
            depth,
            free_depth
        ));

        return data;
    }

    py::list read_orthographic() {
        setOrthographicImageSize(1032, 772);

        py::list data;
        if (!m_running) {
            return data;
        }

        pho::api::PFrame frame = get();
        if (!frame || frame->Empty() || frame->PointCloud.Empty()) {
            return data;
        }


        // Extract texture (grayscale image)
        size_t texture_w = frame->Texture.Size.Width;
        size_t texture_h = frame->Texture.Size.Height;
        float *texture = new float[texture_w * texture_h];
        frame->Texture.ConvertTo2DArray(texture, texture_h, texture_w);
        py::capsule free_texture(texture, [](void *f) {
            float *r = reinterpret_cast<float *>(f);
            delete[] r;
        });
        data.append(py::array_t<float>(
            {texture_h, texture_w},
            {texture_w*sizeof(float), sizeof(float)},
            texture,
            free_texture
        ));

        // Extract depth map
        size_t depth_w = frame->DepthMap.Size.Width;
        size_t depth_h = frame->DepthMap.Size.Height;
        float *depth = new float[depth_w * depth_h];
        frame->DepthMap.ConvertTo2DArray(depth, depth_h, depth_w);
        py::capsule free_depth(depth, [](void *f) {
            float *r = reinterpret_cast<float *>(f);
            delete[] r;
        });
        data.append(py::array_t<float>(
            {depth_h, depth_w},
            {depth_w*sizeof(float), sizeof(float)},
            depth,
            free_depth
        ));


        glEnable(GL_DEPTH_TEST);

        // double pixel_density = 1.0;
        double alpha = 0.4;
        size_t min_depth = 850.0;
        size_t max_depth = 1250.0;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(alpha * width, -alpha * width, -alpha * height, alpha * height, min_depth, max_depth);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(-40.0, -60.0, 0.0, 60.0, -60.0, 1000.0, 0.0, 1000.0, 0.0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
        {
            pho::api::Point3_32f ZeroPoint(0.0f, 0.0f, 0.0f);

            auto ptr = frame->PointCloud.GetDataPtr();
            auto texture = frame->Texture.GetDataPtr();
            for (int y = 0; y < frame->PointCloud.Size.Height; ++y) {
                for (int x = 0; x < frame->PointCloud.Size.Width; ++x) {
                    if (frame->PointCloud[y][x] != ZeroPoint) {
                        size_t i = y * frame->PointCloud.Size.Width + x;
                        
                        float value = *(float*)(texture + i);
                        value = std::pow(value, 1.0/2.2) / 255;
                        glColor3f(value, value, value);

                        glVertex3fv((GLfloat*)&((ptr + i)->x));
                    }
                }
            }
        }
        glEnd();
        glFinish();


        // Extract texture (grayscale image)
        data.append(py::array_t<GLfloat>(
            {height, width, 4},
            {4*width*sizeof(GLfloat), 4*sizeof(GLfloat), sizeof(GLfloat)},
            (const float*)buffer
        ));

        // Extract texture (depth image)
        GLint w, h;
        GLint bytesPerValue;
        void* depthBuffer;
        if (!OSMesaGetDepthBuffer(ctx, &w, &h, &bytesPerValue, &depthBuffer)) {
            std::cout << "OSMesaMakeCurrent get depth!" << std::endl;
        }

        data.append(py::array_t<GLushort>(
            {height, width},
            {width*sizeof(GLushort), sizeof(GLushort)},
            (const short unsigned int*)depthBuffer
        ));

        return data;
    }
};


PYBIND11_MODULE(native_phoxi_driver, m) {
    py::class_<NativePhoXiDriver>(m, "NativePhoXiDriver")
        .def(py::init<const std::string&>())
        .def("start", &NativePhoXiDriver::start)
        .def("stop", &NativePhoXiDriver::stop)
        .def("read", &NativePhoXiDriver::read)
        .def("read_orthographic", &NativePhoXiDriver::read_orthographic);
}
