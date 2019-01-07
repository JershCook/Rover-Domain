"""
Provides Open AI gym wrapper for rover domain simulation core with some extra
    gym-specific functionality. This is the gym equivalent to 'getSim()' in
    the specific.py file.

    Get a default rover domain simulation with some default functionality.
    Users are encouraged to modify this function and save copies of it for
     each trial to use as a parameter reference.

Set data["Reward Function"] to define the reward function callback
Set data["Evaluation Function"] to define the evaluation function callback
Set data["Observation Function"] to define the observation function callback

Note: step function returns result of either the reward or evaluation function
    depending mode ("Train" vs "Test" respectively)

RoverDomainGym should be mods
"""

from core import SimulationCore
import pyximport
import code.world_setup as world_setup  # Rover Domain Construction
import code.agent_domain_2 as rover_domain  # Rover Domain Dynamic
import code.reward_2 as rewards  # Agent Reward and Performance Recording
from code.trajectory_history import *  # Record trajectory of agents for calculating rewards

pyximport.install()  # For cython(pyx) code

import matplotlib.pyplot as plt

class RoverDomainGym(SimulationCore):
    def __init__(self):
        SimulationCore.__init__(self)

        self.data["Number of Agents"] = 30
        self.data["Number of POIs"] = 8
        self.data["Minimum Distance"] = 1.0
        self.data["Steps"] = 100
        self.data["Trains per Episode"] = 50
        self.data["Tests per Episode"] = 1
        self.data["Number of Episodes"] = 5000
        self.data["Specifics Name"] = "test"
        self.data["Mod Name"] = "global"
        self.data["World Index"] = 0

        # Add Rover Domain Construction Functionality
        # Note: reset() will generate random world based on seed
        self.data["World Width"] = 50
        self.data["World Length"] = 50
        self.data['Poi Static Values'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.data['Poi Relative Static Positions'] = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
            [0.5, 0.0]
        ])
        self.data['Agent Initialization Size'] = 0.1
        self.trainBeginFuncCol.append(world_setup.blueprintStatic)
        self.trainBeginFuncCol.append(world_setup.blueprintAgentInitSize)
        self.worldTrainBeginFuncCol.append(world_setup.initWorld)
        self.testBeginFuncCol.append(world_setup.blueprintStatic)
        self.testBeginFuncCol.append(world_setup.blueprintAgentInitSize)
        self.worldTestBeginFuncCol.append(world_setup.initWorld)

        # Add Rover Domain Dynamic Functionality
        """
        step() parameter [action] (2d numpy array with double precision):
            Actions for all rovers before clipping -1 to 1 defined by 
            doAgentMove.
            Dimensions are agentCount by 2.
            
        step()/reset() return [observation] (2d numpy array with double
            precision): Observation for all agents defined by data["Observation 
            Function"].
            Dimensions are agentCount by 8.
            
        For gym compatibility, self.data["Observation Function"] is
        called automatically by this object, no need to call it in a 
        function collection
        """
        self.data["Observation Function"] = rover_domain.doAgentSense
        self.worldTrainStepFuncCol.append(rover_domain.doAgentMove)
        self.worldTestStepFuncCol.append(rover_domain.doAgentMove)

        # Add Agent Training Reward and Evaluation Functionality
        """
        Training Mode:
        step() return [reward] (1d numpy array with double precision): Reward 
            defined by data["Reward Function"]
            Length is agentCount.
            
        Testing Mode:
        step() return [reward] (double): Performance defined by 
            data["Evaluation Function"]
        """
        self.data["Coupling"] = 1
        self.data["Observation Radius"] = 4.0
        self.data["Reward Function"] = rewards.assignGlobalReward
        self.data["Evaluation Function"] = rewards.assignGlobalReward

        self.worldTrainBeginFuncCol.append(createTrajectoryHistories)
        self.worldTrainStepFuncCol.append(updateTrajectoryHistories)
        self.worldTestBeginFuncCol.append(createTrajectoryHistories)
        self.worldTestStepFuncCol.append(updateTrajectoryHistories)

        # TODO make these be hidden class attributes, no reason to have them be lambdas
        # TODO for what should be a fixed-environment scenario
        self.worldTrainBeginFuncCol.append(
            lambda data: data.update({"Gym Reward": np.zeros(data['Number of Agents'])})
        )
        self.worldTestBeginFuncCol.append(
            lambda data: data.update({"Gym Reward": 0})
        )
        self.worldTrainEndFuncCol.append(
            lambda data: data["Reward Function"](data)
        )
        self.worldTrainEndFuncCol.append(
            lambda data: data.update({"Gym Reward": data["Agent Rewards"]})
        )
        self.worldTestEndFuncCol.append(
            lambda data: data["Evaluation Function"](data)
        )
        self.worldTestEndFuncCol.append(
            lambda data: data.update({"Gym Reward": data["Global Reward"]})
        )

        # Setup world for first time
        self.reset(new_mode="Train", fully_resetting=True)

    def step(self, action):
        """
        Proceed 1 time step in world if world is not done
        
        Args:
        action: see rover domain dynamic functionality comments in __init__()
        
        Returns:
        observation: see rover domain dynamic functionality comments in 
            __init__()
        reward: see agent training reward functionality comments for 
            data["Mode"] == "Test" and performance recording functionality 
            comment for data["Mode"] == "Test"
        done (boolean): Describes with the world is done or not
        info (dictionary): The state of the simulation as a dictionary of data
        
        """
        # Store Action for other functions to use
        self.data["Agent Actions"] = action

        # If not done, do step functionality
        if self.data["Step Index"] < self.data["Steps"]:

            # Do Step Functionality
            self.data["Agent Actions"] = action
            if self.data["Mode"] == "Train":
                for func in self.worldTrainStepFuncCol:
                    func(self.data)
            elif self.data["Mode"] == "Test":
                for func in self.worldTestStepFuncCol:
                    func(self.data)
            else:
                raise Exception(
                    'data["Mode"] should be set to "Train" or "Test"'
                )

            # Increment step index for future step() calls
            self.data["Step Index"] += 1
            
            
            
            # Check is world is done; if so, do ending functions
            if self.data["Step Index"] >= self.data["Steps"]:
                if self.data["Mode"] == "Train":
                    for func in self.worldTrainEndFuncCol:
                        func(self.data)
                elif self.data["Mode"] == "Test":
                    for func in self.worldTestEndFuncCol:
                        func(self.data)
                else:
                    raise Exception(
                        'data["Mode"] should be set to "Train" or "Test"'
                    )
                    
            # Observe state, store result in self.data
            self.data["Observation Function"](self.data)


        # Check if simulation is done
        done = False
        if self.data["Step Index"] >= self.data["Steps"]:
            done = True

        return self.data["Agent Observations"], self.data["Gym Reward"], done, self.data

    def reset(self, new_mode=None, fully_resetting=False):
        """
        Reset the world 
            
        Args:
        mode (None, String): Set to "Train" to enable functions associated with 
            training mode. Set to "Test" to enable functions associated with 
            testing mode instead. If None, does not change current simulation 
            mode.
        fully_resetting (boolean): If true, do addition functions
            (self.trainBeginFuncCol) when setting up world. Typically used for
            resetting the world for a different episode and/or different
            training/testing simulation mode.
            
        Returns:
        observation: see rover domain dynamic functionality comments in 
            __init__()
        """
        # Zero step index for future step() calls
        self.data["Step Index"] = 0

        # Set mode if not None
        if new_mode is not None:
            self.data["Mode"] = new_mode

        # Execute setting functionality
        if self.data["Mode"] == "Train":
            if fully_resetting:
                for func in self.trainBeginFuncCol:
                    func(self.data)
            for func in self.worldTrainBeginFuncCol:
                func(self.data)
        elif self.data["Mode"] == "Test":
            if fully_resetting:
                for func in self.testBeginFuncCol:
                    func(self.data)
            for func in self.worldTestBeginFuncCol:
                func(self.data)
        else:
            raise Exception('data["Mode"] should be set to "Train" or "Test"')

        # Observe state, store result in self.data
        self.data["Observation Function"](self.data)

        return self.data["Agent Observations"]
        
    def render(self):
        scale=.5
        nPois= self.data["Number of POIs"]//2
        if (self.data["World Index"] ==0):
            plt.ion()
        plt.clf()
        plt.xlim(-self.data["World Width"]*scale,self.data["World Width"]*(1.0+scale))
        plt.ylim(-self.data["World Length"]*scale,self.data["World Length"]*(1.0+scale))
        
        plt.scatter(self.data["Agent Positions"][:,0],self.data["Agent Positions"][:,1])
        
        if ("Sequential" in self.data and self.data["Sequential"]):
            plt.scatter(self.data["Poi Positions"][nPois:,0],self.data["Poi Positions"][nPois:,1])
            plt.scatter(self.data["Poi Positions"][:nPois,0],self.data["Poi Positions"][:nPois,1])
        else:
            plt.scatter(self.data["Poi Positions"][:,0],self.data["Poi Positions"][:,1])
        
        plt.draw()
        plt.pause(1.0/30.0)
