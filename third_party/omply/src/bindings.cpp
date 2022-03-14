#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <omply/collision_checker.hpp>
#include <omply/dual_arm.hpp>
#include <omply/single_arm.hpp>
#include <trac_ik/trac_ik.hpp>


namespace py = pybind11;
using namespace YuMiPlanning;

PYBIND11_MODULE(_omply, m) {
    py::class_<CollisionChecker>(m, "CollisionChecker")
        .def(py::init<const std::string&>())
        .def("isColliding", &CollisionChecker::isColliding)
        .def("isInBounds", &CollisionChecker::isInBounds)
        .def("getDistance", &CollisionChecker::getDistance);
    
    py::class_<DualArmPlanner>(m, "DualArmPlanner")
        .def(py::init<CollisionChecker>())
        .def("planPath", &DualArmPlanner::planPathPy);
    
    py::class_<SingleArmPlanner>(m, "SingleArmPlanner")
        .def(py::init<CollisionChecker, bool, bool>())
        .def("planPath", &SingleArmPlanner::planPathPy);

    py::enum_<TRAC_IK::SolveType>(m, "SolveType")
        .value("Speed", TRAC_IK::SolveType::Speed)
        .value("Distance", TRAC_IK::SolveType::Distance)
        .value("Manip1", TRAC_IK::SolveType::Manip1)
        .value("Manip2", TRAC_IK::SolveType::Manip2)
        .export_values();

    py::class_<TRAC_IK::TRAC_IK>(m, "TracIK")
        .def(py::init<const std::string&, const std::string&, const std::string&, double, double, TRAC_IK::SolveType>(), py::arg("base_frame"), py::arg("tip_frame"), py::arg("urdf_string"), py::arg("timeout")=0.05, py::arg("eps")=1e-5, py::arg("solve_type")=TRAC_IK::SolveType::Speed)
        .def_property("joint_limits", [](TRAC_IK::TRAC_IK& t) {
            KDL::JntArray l, u;
            t.getKDLLimits(l, u);
            return py::make_tuple(l.data, u.data);
        }, [](TRAC_IK::TRAC_IK& t, py::tuple& tuple) {
            KDL::JntArray l, u;
            l.data = tuple[0].cast<Eigen::VectorXd>();
            u.data = tuple[1].cast<Eigen::VectorXd>();
            t.setKDLLimits(l, u);
        })
        .def("ik", [](TRAC_IK::TRAC_IK& t, Eigen::MatrixXd& pose, Eigen::VectorXd& q_init, Eigen::VectorXd& bounds) -> py::object {
            KDL::JntArray kdl_q_init, result;
            kdl_q_init.data = q_init;

            KDL::Frame kdl_pose {
                KDL::Rotation(pose(0, 0), pose(0, 1), pose(0, 2), pose(1, 0), pose(1, 1), pose(1, 2), pose(2, 0), pose(2, 1), pose(2, 2)),
                KDL::Vector(pose(0, 3), pose(1, 3), pose(2, 3))
            };

            KDL::Twist kdl_bounds {
                KDL::Vector(bounds(0), bounds(1), bounds(2)),
                KDL::Vector(bounds(3), bounds(4), bounds(5))
            };

            int success = t.CartToJnt(kdl_q_init, kdl_pose, result, kdl_bounds);
            if (success < 0) {
                return py::none();
            }
            return py::cast(result.data);
        });
}
