# Load julia packages
using Pkg
Pkg.activate("forward_graphics_engine/halluc_prog")
using Distributions
using ProgressMeter
using Gen
using DataFrames
using CSV

# Set project directory
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"

# Import Gen distribution
include(proj_dir*"/bin/utils/dirichlet.jl")

# Set variables
alpha = repeat([10000], 10) # Doesn't matter - nothing generated randomly

@gen function room()
    # Choose scene
    scenes = ["bathroom", "bedroom", "dining-room", "grey-white-room", "kitchen", "living-room", "staircase", "study", "tea-room"]
    scene ~ Gen.categorical([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9])
    sscene = scenes[scene]
    
    # Choose agent
    agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
    theta = Gen.dirichlet(alpha)
    agent ~ Gen.categorical(theta)
    sagent = agents[agent]

    # Draw position of agent (x, y)
    if sagent == "none"
        # Draw random
        x ~ Gen.uniform(0, 0)
        z ~ Gen.uniform(0, 0)
    else
        # Draw random
        x ~ Gen.uniform(0, 1)
        z ~ Gen.uniform(0, 1)
    end
end

# use the generative model to generate synthetic data
function check_maker()
    df = DataFrame(image=String[], bathroom=Int[], bedroom=Int[], dining_room=Int[], grey_white_room=Int[], kitchen=Int[], living_room=Int[], staircase=Int[], study=Int[], tea_room=Int[], none=Int[], cap=Int[], camera=Int[], boot=Int[], bird=Int[], cat=Int[], dog=Int[], baby=Int[], woman=Int[], man=Int[], x=Float32[], z=Float32[])

    # Define scenes and agents
    scenes = ["bathroom", "bedroom", "dining-room", "grey-white-room", "kitchen", "living-room", "staircase", "study", "tea-room"]
    agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]

    # Create index
    k = 1
    for scene = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for agent = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            for x = [0, 1]
                for z = [0, 1]
		            println("Check #", k)

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

		            # Generate png file name
		            filename = join([sscene, sagent, string(k)*".png"], "_")

		            # Save to dataframe
                    push!(df, [filename, vscene[1], vscene[2], vscene[3], vscene[4], vscene[5], vscene[6], vscene[7], vscene[8], vscene[9], vagent[1], vagent[2], vagent[3], vagent[4], vagent[5], vagent[6], vagent[7], vagent[8], vagent[9], vagent[10], x, z])

		            # Step index
		            k += 1
                end
            end
        end
    end
    println("Writing labels")
    CSV.write(proj_dir*"/images/check/labels.csv", df)

    return nothing
end

println("------------------------------")
if isfile(proj_dir*"/images/check/labels.csv")
    println("CHECK CSV EXISTS")
else
    println("CREATING CSV DATA")
    mkpath(proj_dir*"/images/check/")
    check_maker()
end
