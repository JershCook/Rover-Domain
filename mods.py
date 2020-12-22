import datetime
from code.reward_2 import * # Agent Reward 
from code.curriculum import * # Agent Curriculum
from mod_funcs import * 
from math import sqrt

def globalRewardMod(sim):
    sim.data["Mod Name"] = "global"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardMod(sim):
    sim.data["Mod Name"] = "difference"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)

def dppRewardMod(sim):
    sim.data["Mod Name"] = "dpp"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignDppReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)



def globalRewardSizeCurrMod10(sim):
    sim.data["Schedule"] = ((10.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr10"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardSizeCurrMod20(sim):
    sim.data["Schedule"] = ((20.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr20"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        
def globalRewardSizeCurrMod30(sim):
    sim.data["Schedule"] = ((30.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr30"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardSizeCurrMod40(sim):
    sim.data["Schedule"] = ((40.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr40"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        

def globalRewardCoupCurrMod1(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((1, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr1"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod2(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((2, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr2"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod3(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((3, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr3"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod4(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((4, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr4"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod5(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((5, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr5"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
##################################################################################


        
def differenceRewardSizeCurrMod10(sim):
    sim.data["Schedule"] = ((10.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr10"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardSizeCurrMod20(sim):
    sim.data["Schedule"] = ((20.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr20"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        
def differenceRewardSizeCurrMod30(sim):
    sim.data["Schedule"] = ((30.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr30"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardSizeCurrMod40(sim):
    sim.data["Schedule"] = ((40.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr40"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        

def differenceRewardCoupCurrMod1(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((1, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr1"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod2(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((2, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr2"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod3(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((3, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr3"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod4(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((4, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr4"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod5(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((5, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr5"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        


'''
:param sim:  provides a simulation with the global data structure
:returns: none
:pre: policies have been assigned to each agent
:post: one of the existing policies is reassigned to each agent
:note: call function after sim.reset. data["World Index"] is used to determine which population to use and must also be set
'''           
        
def assignHomogeneousPolicy(sim):
    data=sim.data
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[0][worldIndex]
    data["Agent Policies"] = policyCol 

   
'''
:param sim:  provides a simulation with the global data structure
:returns: none
:pre: none
:post:poi move with a seeded random velocity
:note: call function after sim.step
'''
def poiVelocity(sim):
    data=sim.data
    
    if not "Poi Velocity" in data:
        state=np.random.get_state()
        np.random.seed(123)
        data["Poi Velocity"]=np.random.random(data["Poi Positions"].shape)-.5  
        np.random.set_state(state)   
    data["Poi Positions"]+=data["Poi Velocity"]*0.5
    

'''
:param sim:  provides a simulation with the global data structure
:returns: none
:pre: An action has been determined
:post: agents have varying max speeds from 50% to 100%
:note: call function after actions are determined 
'''
        
def abilityVariation(sim):
    data=sim.data
    
    variation=np.linspace(0.5,1.0, sim.data["Number of Agents"])
    
    for n in range(sim.data["Number of Agents"]):
        sim.data["Agent Actions"][n,:] *= variation[n]  

'''
:param data:  global data structure
:returns: none
:pre: an array holds whether or not an agent has found a "key"
:post: array is cleared, indicating that no agents are holding "keys"
:note: none 
'''    

def clearItemHeld(data):
    nAgents=sim.data["Number of Agents"]
    sim.data["Item Held"] =np.zeros((nAgents), dtype = np.int32)
    

'''
:param sim:  Provides a simulation with the global data structure
:returns: None
:pre: None
:post: Agents must go to poi type-a to recieve a "key" and then group at poi type-b to open the "lock" and recieve a reward. Poi[0:n/2] = Type B and Poi[n/2:n] = Type A
:note: Call function after sim is created 
'''

def sequentialPoi(sim):

    sim.data["Sequential"]=True
    sim.data["Observation Function"]=doAgentSenseMod
    
    sim.data["Reward Function"]=assignGlobalRewardMod
    
    sim.worldTrainBeginFuncCol.append(  clearItemHeld  )
    
    if not "View Distance" in sim.data: sim.data["View Distance"]= -1
    


'''
:param sim: Provides a simulation with the global data structure
:returns: None
:pre: A visibility range is given to each agent
:post: Agents can only perceive items in the visibility range
:note: Call function after sim is created 
'''
    
def lowVisibility(sim):
    if not "Sequential" in sim.data: sim.data["Sequential"]= False
    
    sim.data["View Distance"]=15    
	
    sim.data["Observation Function"]=doAgentSenseMod

'''
:param data:  Global data structure
:returns: None
:pre: One step of the simulation has passed
:post: Rewards are assigned based on the number of total parts of the recipe completed for each agent. The max reward is: Number_of_Agents * Size_of_Recipe
:note: None 
''' 

def simpleReward(data):
    number_agents=data["Number of Agents"]

    #globalReward=np.sum(data["Item Held"])
    d=data["Item Held"]
    globalReward=sum(d[0])/len(d[0])
    #d=np.sum(d[:,-4:],axis=0)
    #if data["Global Recipe"]:
    #    d*=number_agents
    #globalReward=d[0]*1.0+d[1]*1.33+d[2]*1.66+d[3]*2.0
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = np.ones(number_agents) * globalReward

'''
:param data:  Global data structure
:returns: None
:pre: An array holds whether or not an agent has complete a part of the recipe
:post: Array is cleared, indicating that no agents have completed any part of the recipe
:note: None 
''' 

def resetItemHeld(data):
    nAgents =    data["Number of Agents"]
    recipeSize = data["Recipe Size" ]
    data["Item Held"] = np.zeros(( nAgents,recipeSize), dtype = np.int32)  
    

'''
:param sim:  Provides a simulation with the global data structure
:returns: None
:pre: A recipe of POI types is given to the agent.
:post:  The agents must go to each poi on the list to recieve a reward. The global reward is determined by the number of agents which complete the recipe
:note: Call function after sim is created. Recipe completion can be ordered or unordered 
'''

def recipePoi(sim):

    
    sim.data["Observation Function"]=doAgentSenseRecipe2
    #sim.data["Reward Function"]=assignGlobalRewardSimple  #reward for each recipe completed
    sim.data["Reward Function"]=simpleReward              #reward for each step of recipe completed 
    
    sim.data["Recipe"] = np.array([0,1,2,3],dtype=np.int32) #recipe, each item is a POI type from 0 to (N-Poi Types)-1 
    sim.data["Recipe Size"]=len(sim.data["Recipe"])
    sim.data["Ordered"] = False                              #flag for whether order matters
    sim.data["Number of POI Types"] = 4
    sim.data["Coupling Limit"]=15                            #max number of agents which can see view a poi at a time 
    sim.data["Global Recipe"]=True
    sim.worldTrainBeginFuncCol.append(  resetItemHeld  )

def multiReward(sim):
    
    data=sim.data
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    
    historyStepCount = data["Steps"]
    #coupling = data["Coupling"]
    observationRadiusSqr = data["Observation Radius"] ** 2
    agentPositionHistory = data["Agent Position History"]
    poiValueCol = data['Poi Values']
    poiPositionCol = data["Poi Positions"]
  
    
    #recipe = data["Recipe"]
    #recipeSize = data["Recipe Size"]
    nPoiTypes  = data["Number of POI Types"]
    #ordered    = data["Ordered"] 
    
    Inf = float("inf")
    
    
    rewards=[0.0 for i in range(nPoiTypes + 1)]
 
    
    for poiIndex in range(number_pois):
        poiType = poiIndex % nPoiTypes
        
    
        
        stepIndex = historyStepCount
            
            
        for agentIndex in range(number_agents):
            # Calculate separation distance between poi and agent
            separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
            separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
            distanceSqr = separation0 * separation0 + separation1 * separation1
            
            if distanceSqr < observationRadiusSqr:
                rewards[poiType]+= 1.0/float(number_agents)
            #rewards[poiType]+=-sqrt(distanceSqr)
            dist=0.0
            if poiIndex == 0:
                min_dist=1e9
                
                for otherIndex in range(number_agents):
                    # Calculate separation distance between poi and agent
                    separation0 = agentPositionHistory[stepIndex, otherIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                    separation1 = agentPositionHistory[stepIndex, otherIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                    distanceSqr = separation0 * separation0 + separation1 * separation1
                    
                    dist=sqrt( distanceSqr )
                    if dist<min_dist and dist>0:
                        min_dist=dist
                            
                rewards[-1]+= -min_dist    
                
    return rewards


def posInit(data,mu,sig):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    agentInitSize = sig
    
    worldSize = np.array([world_width, world_length])
    
    # Initialize all agents in the np.randomly in world
    positionCol = np.random.rand(number_agents, 2)-0.5 
    positionCol *= agentInitSize
    positionCol +=0.5 + (np.random.rand(2)-0.5) * mu 

    positionCol *= worldSize
    data['Agent Positions BluePrint'] = positionCol
    angleCol = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angleCol), np.sin(angleCol))).T
