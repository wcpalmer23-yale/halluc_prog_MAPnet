# Load julia packages
using Pkg
# Pkg.activate("halluc_prog")
# Pkg.add(["Distributions",
#         "ProgressMeter",
#         "Gen",
#         "Plots",
#         "Parameters",
#         "DataFrames",
#         "CSV",
#         "PyCall",
#         "Conda"])
#         "ArgParse"])
using Distributions
using ProgressMeter
using Gen, Plots
using DataFrames
using CSV
using PyCall
using Conda
using ArgParse

# Download python packages
# Conda.add("numpy")
# Conda.pip_interop(true)
# Conda.pip("install", "mitsuba")

# Import python packages/scripts
@pyinclude("gen_utils.py")

# Set project directory
proj_dir = "/home/wcp27/projects/halluc_prog_MAPnet"

# Set variables
alpha = repeat([10000], 6) # Doesn't matter - nothing generated randomly

println("Creating clean test images")

@gen function room()
    # Choose scene
    scenes = ["living-room", "bedroom"]
    scene ~ Gen.categorical([0.5, 0.5])
    sscene = scenes[scene]
    
    # Choose agent
    agents = ["none", "cat", "dog", "man", "woman", "baby"]
    theta = Gen.dirichlet(alpha)
    agent ~ Gen.categorical(theta)
    sagent = agents[agent]

    # Draw position of agent (x, y)
    if sagent == "none"
        # Draw random
        x ~ Gen.uniform(0, 0)
        z ~ Gen.uniform(0, 0)
        x_scale = x
        z_scale = z
        y_scale = 0
    else
        # Draw random
        x ~ Gen.uniform(0, 1)
        z ~ Gen.uniform(0, 1)

        # Scale
        if sscene == "living-room"
            x_scale = x*(1.3+1.75)-1.75
            z_scale = z*(1.3+0.75)-0.75
            y_scale = 0
        elseif sscene == "bedroom"
            x_scale = x*(1.5+1)-1
            z_scale = z*(1.5+0.5)-0.5

            if sagent!="man" && sagent!="woman" && x_scale<=0.5 && x_scale>=-1 && z_scale<=0.2 && z_scale>=-0.5
                y_scale = 0.4
            else
                y_scale = 0
            end
        end
    end
    
    # Instantiate Scene
    scene = py"load_scene"(sscene, sagent)::PyObject
    
    # Translate agent
    scene = py"transform_scene"(scene, sagent, x_scale, z_scale, y_scale)::PyObject

    # Scene to Float Array
    mu = py"scene_to_float"(scene, 512)::Array{Float32, 3}

    # define the likelihood as a gaussian with diagonal covariance
    pred ~ broadcasted_normal(mu, 0.05)
end

# use the generative model to generate synthetic data
function test_maker()
    df = DataFrame(image=String[], living_room=Int[], bedroom=Int[], none=Int[], cat=Int[], dog=Int[], man=Int[], 
    woman=Int[], baby=Int[], x=Float32[], z=Float32[])

    # Define scenes and agents
    scenes = ["living-room", "bedroom"]
    agents = ["none", "cat", "dog", "man", "woman", "baby"]

    # Create index
    k = 1
    for scene = [1, 2] # living-room, bedroom
        for agent = [1, 2, 3, 4, 5, 6] #none, cat, dog, man, woman, baby 
            for x = LinRange(0.01, 0.99, 10)
                for z = LinRange(0.01, 0.99, 10)
                    println("Test #", k)
                    # Create constraint
                    constraint = choicemap()
                    constraint[:scene] = scene
                    constraint[:agent] = agent
                    
                    if agent == 1
                        x = 0
                        z = 0
                    end
                    constraint[:x] = x
                    constraint[:z] = z
                    
                    # simulate data using the prior 
                    tr = first(Gen.generate(room, (), constraint))
        
                    # Extract information
                    vscene = zeros(Int8, length(scenes))
                    vscene[scene] = 1
                    sscene = scenes[scene]
            
                    vagent = zeros(Int8, length(agents))
                    vagent[agent] = 1
                    sagent = agents[agent]

                    # Save room as png
                    # Pull prediction scene
                    pred = tr[:pred]

                    # Save pred
                    filename = join([sscene, sagent, string(k)*".png"], '_')
                    py"array_png"(pred, proj_dir*"/images/test/clean/", filename)

                    # Save to dataframe
                    push!(df, [filename, vscene[1], vscene[2], vagent[1], vagent[2], vagent[3], vagent[4], vagent[5], vagent[6], x, z])

                    # Step index
                    k += 1
                end
            end
        end
    end
    CSV.write(proj_dir*"/images/test/labels.csv", df)

    return nothing
end
println("------------------------------")
println("CREATING CLEAN TEST DATA")
test_maker()
