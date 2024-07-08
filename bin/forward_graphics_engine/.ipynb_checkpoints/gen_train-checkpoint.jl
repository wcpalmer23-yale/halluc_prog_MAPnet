# Load julia packages
using Pkg
Pkg.activate("forward_graphics_engine/halluc_prog")
using Distributions
using Gen
using ProgressMeter
using DataFrames
using CSV
using ArgParse

# Set project directory
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"

# Import Gen distribution
include(proj_dir*"/bin/utils/dirichlet.jl")

# Arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--dataset", "-d"
            help = "name of dataset"
            arg_type = String
            required = true
        "--n_train", "-n"
            help = "number of training images"
            arg_type = Int
            required = true
        "--alpha", "-a"
            help = "alphas for dirichlet distributions"
            arg_type = String
            required = true
        "--count", "-c"
            help = "count of previous model predictions"
            arg_type = String
            required = true
    end

    return parse_args(s)
end
parsed_args = parse_commandline()

# Extract Arguments
dataset = parsed_args["dataset"]
n_train = parsed_args["n_train"]
alpha = parse.(Int, split(chop(parsed_args["alpha"]; head=1, tail=1), ','))
count = parse.(Int, split(chop(parsed_args["count"]; head=1, tail=1), ','))
alpha = alpha + count

# Set variables
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
function data_maker(n_train)
    df = DataFrame(image=String[], bathroom=Int[], bedroom=Int[], dining_room=Int[], grey_white_room=Int[], kitchen=Int[], living_room=Int[], staircase=Int[], study=Int[], tea_room=Int[], none=Int[], cap=Int[], camera=Int[], boot=Int[], bird=Int[], cat=Int[], dog=Int[], baby=Int[], woman=Int[], man=Int[], x=Float32[], z=Float32[])

    @showprogress for k = 1:n_train
        # simulate data using the prior 
        tr = Gen.simulate(room, ())
        
        # Extract information
        scenes = ["bathroom", "bedroom", "dining-room", "grey-white-room", "kitchen", "living-room", "staircase", "study", "tea-room"]
        scene = tr[:scene]
        sscene = scenes[scene]

        vscene = zeros(Int8, length(scenes))
        vscene[scene] = 1
            
        agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
        agent = tr[:agent]
        sagent = agents[agent]

        vagent = zeros(Int8, length(agents))
        vagent[agent] = 1

        x = tr[:x]
        z = tr[:z]

        # Generate file name
        filename = join([sscene, sagent, string(k)*".png"], '_')

        # Save to dataframe
        push!(df, [filename, vscene[1], vscene[2], vscene[3], vscene[4], vscene[5], vscene[6], vscene[7], vscene[8], vscene[9], vagent[1], vagent[2], vagent[3], vagent[4], vagent[5], vagent[6], vagent[7], vagent[8], vagent[9], vagent[10], x, z])

        # Save dataframe
        CSV.write(proj_dir*"/images/"*dataset*"/labels.csv", df)
    end
    return nothing
end

println("------------------------------")
if isfile(proj_dir*"/images/"*dataset*"/labels.csv")
    println("TRAINING DATA EXISTS")
    println("Dataset: ", dataset)
    println("Alpha: ", alpha)
else
    # Create data
    println("CREATING TRAINING DATA")
    println("Dataset: ", dataset)
    println("Alpha: ", alpha)
    mkpath(proj_dir*"/images/"*dataset)
    data_maker(n_train)

    # Save alpha to textfile
    open(proj_dir*"/images/"*dataset*"/"*dataset*"_alpha.txt", "w") do file
        write(file, string(alpha))
    end;
end
