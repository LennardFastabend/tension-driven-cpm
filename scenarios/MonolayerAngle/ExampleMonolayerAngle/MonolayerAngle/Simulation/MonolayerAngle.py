SimID = 1
NO = 10
xdim = 600
ydim = 600
T = 1
lambda_v = 1 
lambda_s = 1
J10 = 250
J11 = 0
J12 = -250
steps = 15001

A_i = 400
sigma = 1
epsilon_P = 1.273
GT = 0.05
GF = 0.05
dt = 10



def configure_simulation(SimID, NO, xdim, ydim, Temp, J10, J11, J12, steps):

    from cc3d.core.XMLUtils import ElementCC3D

    CompuCell3DElmnt=ElementCC3D("CompuCell3D",{"Revision":"0","Version":"4.3.2"})

    MetadataElmnt=CompuCell3DElmnt.ElementCC3D("Metadata")

    # Basic properties simulation
    MetadataElmnt.ElementCC3D("NumberOfProcessors",{},"1")
    MetadataElmnt.ElementCC3D("DebugOutputFrequency",{},"10")
    # MetadataElmnt.ElementCC3D("NonParallelModule",{"Name":"Potts"})

    PottsElmnt=CompuCell3DElmnt.ElementCC3D("Potts")

    # Basic properties of CPM (GGH) algorithm
    PottsElmnt.ElementCC3D("Dimensions",{"x":str(xdim),"y":str(ydim),"z":"1"})
    PottsElmnt.ElementCC3D("Steps",{},str(steps))
    PottsElmnt.ElementCC3D("Temperature",{},str(Temp))
    PottsElmnt.ElementCC3D("NeighborOrder",{},str(NO))
    PottsElmnt.ElementCC3D("RandomSeed",{},str(SimID))

    PluginElmnt=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CellType"})

    # Listing all cell types in the simulation
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"0","TypeName":"Medium"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"1","TypeName":"Cell"})
    PluginElmnt.ElementCC3D("CellType",{"Freeze":"","TypeId":"2","TypeName":"Wall"})

    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Volume"})

    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Surface"})

    PluginElmnt_1=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"PixelTracker"})

    # Module tracking pixels of each cell

    PluginElmnt_2=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"BoundaryPixelTracker"})

    # Module tracking boundary pixels of each cell
    PluginElmnt_2.ElementCC3D("NeighborOrder",{},"1")

    PluginElmnt_3=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Contact"})
    # Specification of adhesion energies
    PluginElmnt_3.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Cell"},str(J10))
    PluginElmnt_3.ElementCC3D("Energy",{"Type1":"Cell","Type2":"Cell"},str(J11))
    PluginElmnt_3.ElementCC3D("Energy",{"Type1":"Cell","Type2":"Wall"},str(J12))
    PluginElmnt_3.ElementCC3D("NeighborOrder",{},str(NO))



    SteppableElmnt=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"PIFInitializer"})

    # Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator
    SteppableElmnt.ElementCC3D("PIFName",{},"Simulation/rect_600.piff")


    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)



from cc3d import CompuCellSetup


configure_simulation(SimID, NO, xdim, ydim, T, J10, J11, J12, steps)


from MonolayerAngleSteppables import GrowthSteppable
CompuCellSetup.register_steppable(steppable=GrowthSteppable(lambda_v, lambda_s, A_i, sigma, epsilon_P, GT, GF, dt, frequency=1))

from MonolayerAngleSteppables import MitosisSteppable
CompuCellSetup.register_steppable(steppable=MitosisSteppable(A_i, sigma, epsilon_P, frequency=1))

from MonolayerAngleSteppables import TissueTrackerSteppable
CompuCellSetup.register_steppable(steppable=TissueTrackerSteppable(SimID, frequency=100))

from MonolayerAngleSteppables import CellTrackerSteppable
CompuCellSetup.register_steppable(steppable=CellTrackerSteppable(SimID, frequency=100))

from MonolayerAngleSteppables import MacroscopicVolumeTrackerSteppable
CompuCellSetup.register_steppable(steppable=MacroscopicVolumeTrackerSteppable(SimID, frequency=10))


CompuCellSetup.run()
