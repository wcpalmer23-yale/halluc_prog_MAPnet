# Create Julia environment
using Pkg
Pkg.activate("halluc_prog")
Pkg.add(["Distributions",
        "ProgressMeter",
        "Gen",
        "DataFrames",
        "CSV", 
        "ArgParse"])
