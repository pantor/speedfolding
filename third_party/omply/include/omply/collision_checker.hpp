#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include <fcl/BVH/BVH_model.h>
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/broadphase/broadphase_bruteforce.h>
#include <fcl/collision.h>
#include <fcl/distance.h>
#include <fcl/shape/geometric_shapes.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <urdf/model.h>

#include <stl/stl_reader.h>


namespace YuMiPlanning {

using Model = fcl::BVHModel<fcl::OBBRSS>;
using ColObjPtr = std::shared_ptr<fcl::CollisionObject>;
using LinkPtr = urdf::LinkSharedPtr;

class CollisionChecker {
    void setupManager(std::shared_ptr<fcl::BroadPhaseCollisionManager> man, const std::string& tip_frame) {
        KDL::Chain chain;
        tree.getChain(BASE, tip_frame, chain);
        for (int i=0; i<chain.getNrOfSegments(); i++) {
            const std::string name = chain.getSegment(i).getName();
            auto obj = objects.find(name);
            if (obj != objects.end()) {
                man->registerObject((*obj).second.get());
            }
        }
        man->setup();
    }

    std::vector<urdf::JointSharedPtr> getJoints(const std::string& tip_frame) const {
        urdf::LinkConstSharedPtr tmp = robot_model.getLink(tip_frame);
        std::vector<urdf::JointSharedPtr> js(7);
        int i = 6;
        while (tmp->name != BASE) {
            urdf::JointSharedPtr j = tmp->parent_joint;
            if (j->type == urdf::Joint::REVOLUTE) {
                js[i--] = j;
            }
            tmp = robot_model.getLink(tmp->parent_joint->parent_link_name);
        }
        return js;
    }

    ColObjPtr loadMesh(const std::string& filename) const {
        std::vector<fcl::Triangle> tris;
        std::vector<fcl::Vec3f> verts;
        
        try {
            stl_reader::StlMesh <float, unsigned int> mesh (filename);
            verts.resize(mesh.num_vrts());
            for (int verti=0;verti<mesh.num_vrts();verti++) {
                const float* c = mesh.vrt_coords(verti);
                verts[verti] = INFLATE*fcl::Vec3f(c[0],c[1],c[2]);
            }
            for (size_t itri = 0; itri < mesh.num_tris(); ++itri) {
                tris.emplace_back(mesh.tri_corner_ind(itri,0),mesh.tri_corner_ind(itri,1),mesh.tri_corner_ind(itri,2));
            }
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        auto geom = std::make_shared<Model>();
        geom->beginModel();
        geom->addSubModel(verts,tris);
        geom->endModel();
        return std::make_shared<fcl::CollisionObject>(geom);
    }

    KDL::Frame getFK(const std::string& frame_name, const std::vector<double>& jointvec) const {
        KDL::Chain chain;
        tree.getChain(BASE, frame_name, chain);
        KDL::ChainFkSolverPos_recursive solver(chain);
        const size_t n_joints = chain.getNrOfJoints();

        KDL::JntArray kdlJ(n_joints);
        for (size_t i = 0; i < n_joints; i++) {
            kdlJ(i) = jointvec[i];
        }

        KDL::Frame result;
        solver.JntToCart(kdlJ, result);
        return result;
    }

    const double INFLATE = 1.11; // all link meshes are slightly blown up to give the yumi controller enough buffer
    urdf::Model robot_model;
    KDL::Tree tree;

    std::unordered_map<std::string, ColObjPtr> objects; // map that keeps track of all the loaded meshes
    
    ColObjPtr tableObj, camObj;
    std::shared_ptr<fcl::DynamicAABBTreeCollisionManager> l_manager, r_manager;
    std::vector<urdf::JointSharedPtr> l_joints, r_joints;

public:
    const std::string L_TIP = "gripper_l_finger_l"; // TODO ideally gripper collisions would be approximated as a larger box
    const std::string R_TIP = "gripper_r_finger_r";
    const std::string BASE = "base_link";

    CollisionChecker(const std::string& desc_path) {
        robot_model.initFile(desc_path + "/urdf/yumi.urdf");

        if (!kdl_parser::treeFromUrdfModel(robot_model, tree)) {
            std::cerr << "Failed to extract kdl tree from xml robot description\n";
        }

        std::vector<LinkPtr> links;
        robot_model.getLinks(links);
        for (auto l: links) {
            if (l->collision != nullptr) {
                const std::string frame_name = l->name;
                
                if (l->collision->geometry->type == urdf::Geometry::MESH) {
                    auto mesh = (urdf::Mesh*)(l->collision->geometry.get());
                    const size_t i = mesh->filename.find("meshes");
                    objects[frame_name] = loadMesh(desc_path + mesh->filename.substr(i));
                }
            }
        }

        //initialize the table object
        const double h = 0.5;
        const double table_z = -0.02;
        std::shared_ptr<fcl::Box> table=std::make_shared<fcl::Box>(2,2,.5);
        const fcl::Vec3f tableT(0,0,-h/2 + table_z);
        tableObj = std::make_shared<fcl::CollisionObject>(table);
        tableObj->setTranslation(tableT);
        
        //initialize the camera object
        const double wx = 0.2;
        const double wy = 0.31;
        const double wz = 0.2;
        std::shared_ptr<fcl::Box> cam = std::make_shared<fcl::Box>(wx,wy,wz);
        fcl::Vec3f camT(0.318598, -0.087999, 0.826867+wz/2);
        camObj = std::make_shared<fcl::CollisionObject>(cam);
        camObj->setTranslation(camT);

        //initialize joints from urdf for bounds
        l_joints = getJoints(L_TIP);
        r_joints = getJoints(R_TIP);

        //initialize the broadphase collision managers
        l_manager = std::make_shared<fcl::DynamicAABBTreeCollisionManager>();
        r_manager = std::make_shared<fcl::DynamicAABBTreeCollisionManager>();
        setupManager(l_manager,L_TIP);
        setupManager(r_manager,R_TIP);
    }

    void getLeftJointLims(double *lower, double *upper) const {
        for (int i=0; i<7; i++) {
            lower[i] = l_joints[i]->limits->lower;
            upper[i] = l_joints[i]->limits->upper;
        }
    }

    void getRightJointLims(double *lower, double *upper) const {
        for (int i=0; i<7; i++) {
            lower[i] = r_joints[i]->limits->lower;
            upper[i] = r_joints[i]->limits->upper;
        }
    }

    bool isInBounds(const std::vector<double>& l_current_joints, const std::vector<double>& r_current_joints) const {
        for (int i=0; i<7; i++) {
            if (l_current_joints[i] < l_joints[i]->limits->lower || l_current_joints[i] > l_joints[i]->limits->upper) {
                return false;
            }

            if (r_current_joints[i] < r_joints[i]->limits->lower || r_current_joints[i] > r_joints[i]->limits->upper) {
                return false;
            }
        }
        return true;
    }

    // returns true if the arm from base link to tip frame is colliding with itself
    bool isSelfColliding(const std::vector<double>& joints, const std::string& tip_frame) {
        KDL::Chain chain;
        tree.getChain(BASE, tip_frame, chain);
        std::vector<ColObjPtr> objsToCheck; // in order from base to elbow
        objsToCheck.push_back(objects[BASE]);
        for (size_t i = 0; i < chain.getNrOfSegments(); i++) {
            const std::string name = chain.getSegment(i).getName();
            auto obj = objects.find(name);
            if (obj != objects.end()) {
                ColObjPtr& o = (*obj).second;
                KDL::Frame H = getFK(name, joints); // TODO optimize this to not be a separate function?
                fcl::Matrix3f rot(H(0,0), H(0,1), H(0,2),
                                  H(1,0), H(1,1), H(1,2),
                                  H(2,0), H(2,1), H(2,2));
                fcl::Vec3f trans(H(0,3), H(1,3), H(2,3));
                o->setTransform(rot, trans);
                objsToCheck.push_back(o);
            }
        }
        for (size_t obj1 = 0; obj1 < objsToCheck.size(); obj1++) {
            for (size_t obj2 = 0; obj2 < objsToCheck.size(); obj2++) {
                if( (obj1-obj2<=1 && obj1-obj2>=-1) || (obj1>=5 && obj2>=5)) {
                    continue;
                }
                //skip neighbor links and links that are past the wrist (these are impossible to collide)
                fcl::CollisionRequest req;
                fcl::CollisionResult res;
                fcl::collide(objsToCheck[obj1].get(), objsToCheck[obj2].get(), req, res);
                if (res.isCollision()) {
                    return true;
                }
            }
        }
        return false;
    }

    // return true if the arms collide with the table (or camera)
    bool environCollision() const {
        int collision = 0;
        auto col_cb = [](fcl::CollisionObject *o1, fcl::CollisionObject *o2, void* dat){
            int *col = (int*)dat;
            fcl::CollisionRequest req;
            fcl::CollisionResult res;
            fcl::collide(o1,o2, req, res);
            if (res.isCollision()) {
                *col=1;
            }
            return res.isCollision();
        };
        l_manager->collide(tableObj.get(), &collision, col_cb);
        if (collision==1) return true;
        r_manager->collide(tableObj.get(), &collision, col_cb);
        if (collision==1) return true;
        l_manager->collide(camObj.get(), &collision, col_cb);
        if (collision==1) return true;
        r_manager->collide(camObj.get(), &collision, col_cb);
        return collision == 1;
    }

    // returns true if the left and right arm collide
    bool interCollision() const {
        int data = 0;
        l_manager->collide(r_manager.get(), &data, [](fcl::CollisionObject *o1, fcl::CollisionObject *o2, void* dat) {
            int *col = (int*)dat;
            fcl::CollisionRequest req;
            fcl::CollisionResult res;
            fcl::collide(o1, o2, req, res);
            if (res.isCollision()) {
                *col = 1;
            }
            return res.isCollision();
        });
        return data == 1;
    }

    double interDistance() const {
        double data, dist;
        l_manager->distance(r_manager.get(), &data, [](fcl::CollisionObject *o1, fcl::CollisionObject *o2, void* dat, double& dist) {
            double *col = (double*)dat;
            fcl::DistanceRequest req;
            fcl::DistanceResult res;
            fcl::distance(o1, o2, req, res);
            *col = res.min_distance;
            return true;
        });
        return data;
    }

    // returns true if the arm is colliding with itself
    // IMPORTANT: the intercollision must happen after both isColliding, because that sets the transforms for all the objects
    bool isColliding(const std::vector<double>& l_joints, const std::vector<double>& r_joints) {
        const bool intraarm = isSelfColliding(l_joints, L_TIP) || isSelfColliding(r_joints, R_TIP); 
        if (intraarm) {
            return true;
        }
        l_manager->update();
        r_manager->update();
        return environCollision() || interCollision();
    }

    double getDistance(const std::vector<double>& l_joints, const std::vector<double>& r_joints) {
        // Why update before and after?
        l_manager->update();
        r_manager->update();
        const bool intraarm = isSelfColliding(l_joints, L_TIP) || isSelfColliding(r_joints, R_TIP); 
        l_manager->update();
        r_manager->update();
        return interDistance();
    }
};

} // namespace YuMiPlanning
