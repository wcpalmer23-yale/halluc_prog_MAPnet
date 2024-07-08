# Load julia packages
using Pkg
Pkg.activate("halluc_prog")
using Distributions
using ProgressMeter
using Gen
using DataFrames
using CSV

# Set project directory
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"

# Set variables
alpha = repeat([10000], 6) # Doesn't matter - nothing generated randomly

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
    else
        # Draw random
        x ~ Gen.uniform(0, 1)
        z ~ Gen.uniform(0, 1)
    end
end

# use the generative model to generate synthetic data
function test_maker()
    df = DataFrame(image=String[], living_room=Int[], bedroom=Int[], none=Int[], cat=Int[], dog=Int[], man=Int[], woman=Int[], baby=Int[], x=Float32[], z=Float32[])

    # Define scenes and agents
    scenes = ["living-room", "bedroom"]
    agents = ["none", "cat", "dog", "man", "woman", "baby"]

    # Create index
    k = 1
    for scene = [1, 2] # living-room, bedroom
        for agent = [1, 2, 3, 4, 5, 6] #none, cat, dog, man, woman, baby 
            for x = LinRange(0.01, 0.99, 4)
                for z = LinRange(0.01, 0.99, 4)
		            println("Example #", k)

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
                    push!(df, [filename, vscene[1], vscene[2], vagent[1], vagent[2], vagent[3], vagent[4], vagent[5], vagent[6], x, z])

		            # Step index
		            k += 1
                end
            end
        end
    end
    println("Writing labels")
    CSV.write(proj_dir*"/images/examples/labels.csv", df)

    return nothing
end

println("------------------------------")
if isfile(proj_dir*"/images/examples/labels.csv")
    println("TEST CSV EXISTS")
else
    println("CREATING CSV DATA")
    mkpath(proj_dir*"/images/examples/")
    test_maker()
end
