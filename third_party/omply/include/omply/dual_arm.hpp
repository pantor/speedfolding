#pragma once

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <omply/collision_checker.hpp>


namespace YuMiPlanning {

class DualArmValidity {
    CollisionChecker colChecker;

public:
    DualArmValidity(CollisionChecker checker): colChecker(checker) {}

    bool isValid(const ompl::base::State *state){
        auto q = state->as<ompl::base::RealVectorStateSpace::StateType>();
        std::vector<double> q_l(7);
        std::vector<double> q_r(7);
        for (int i=0;i<7;i++) q_l[i] = (*q)[i];
        for (int i=0;i<7;i++) q_r[i] = (*q)[i+7];
        return !colChecker.isColliding(q_l, q_r);
    }
};


class DualArmPlanner {
    ompl::geometric::SimpleSetupPtr setup;
    std::shared_ptr<DualArmValidity> validity;

    ompl::base::ScopedState<ompl::base::RealVectorStateSpace> getState(const std::vector<double>& q, ompl::geometric::SimpleSetupPtr setup) {
        ompl::base::ScopedState<ompl::base::RealVectorStateSpace> state(setup->getSpaceInformation());
        const ompl::base::RealVectorBounds &bounds = setup->getStateSpace()->as<ompl::base::RealVectorStateSpace>()->getBounds();
        for (size_t i= 0 ; i < 14; i++) {
            state[i] = std::min(std::max(q[i], bounds.low[i]), bounds.high[i]);
        }
        return state;
    }

public:
    DualArmPlanner(CollisionChecker checker) {
        //initialize the state space
        ompl::base::StateSpacePtr space(new ompl::base::RealVectorStateSpace(14));
        ompl::base::RealVectorBounds bounds(14);
        checker.getLeftJointLims(bounds.low.data(), bounds.high.data());
        checker.getRightJointLims(bounds.low.data()+7, bounds.high.data()+7);
        space->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);
        //use simplesetup to make a planner
        setup = ompl::geometric::SimpleSetupPtr(new ompl::geometric::SimpleSetup(space));
        validity = std::make_shared<DualArmValidity>(checker);
        std::function<bool(const ompl::base::State*)> fun;
        fun = std::bind<bool>(&DualArmValidity::isValid, validity, std::placeholders::_1);
        setup->setStateValidityChecker(fun);
        setup->getSpaceInformation()->setStateValidityCheckingResolution(0.004);//This is fraction of state space, not radians
        ompl::base::PlannerPtr planner(new ompl::geometric::RRTConnect(setup->getSpaceInformation()));
        planner->as<ompl::geometric::RRTConnect>()->setRange(.1);
        setup->setPlanner(planner);
    }

    ompl::geometric::PathGeometric planPath(const std::vector<double>& s, const std::vector<double>& g, double timeout) {
        //execute the solve
        auto start=getState(s, setup);
        auto goal=getState(g, setup);
        setup->clear();
        setup->setStartState(start);
        setup->setGoalState(goal);
        ompl::base::PlannerData dat(setup->getSpaceInformation());
	    int tries=0;
        while (tries++<4) {
            setup->solve(timeout);
            setup->getPlanner()->as<ompl::geometric::RRTConnect>()->getPlannerData(dat);
            if(dat.getGoalIndex(0) != ompl::base::PlannerData::INVALID_INDEX){
                break;
            }
            setup->clear();
            std::cout << "You can safely ignore the above error message\n";
        }
        setup->simplifySolution();
        if (setup->haveExactSolutionPath()) {
            std::cout << "Solution cost "<< setup->getSolutionPath().length() << std::endl;
            return setup->getSolutionPath();
        }
        return ompl::geometric::PathGeometric(setup->getSpaceInformation());
    }

    std::vector<std::vector<double>> planPathPy(const std::vector<double>& s, const std::vector<double>& g, double timeout) {
        ompl::geometric::PathGeometric path = planPath(s, g, timeout);

        std::vector<std::vector<double>> result (path.getStateCount());
        for (size_t s=0; s<path.getStateCount(); s++) {
            std::vector<double> q(14);
            for(size_t i=0; i<14; i++) q[i] = (*(path.getState(s)->as<ompl::base::RealVectorStateSpace::StateType>()))[i];
            result[s] = q;
        }
        return result;
    }
};

} // namespace YuMiPlanning
