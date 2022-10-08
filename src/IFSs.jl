# IFS structs and basic utilities.
module IFSs
using Distributions

export IFS, WIFS, IFSNonlinear
export itrPtsProb

## structs
struct IFS
	linear::Vector{Matrix{Float64}}
	trans::Vector{Vector{Float64}}
	numMaps::Int
	dimAmbient::Int

	IFS(linear::Vector{Matrix{Float64}}, trans::Vector{Vector{Float64}}, numMaps::Int, dimAmbient::Int) = new(linear, trans, numMaps, dimAmbient)
	IFS(linear::Vector{Matrix{Float64}}, trans::Vector{Vector{Float64}}) = IFS(linear, trans, size(trans, 1), size(trans[1], 1))

end

mutable struct WIFS
	linear::Vector{Matrix{Float64}}
	trans::Vector{Vector{Float64}}
	weights::Vector
	numMaps::Int
	dimAmbient::Int

	WIFS(linear::Vector{Matrix{Float64}}, trans::Vector{Vector{Float64}}, weights::Vector) = isprobvec(weights) ? new(linear, trans, weights, size(trans, 1), size(trans[1], 1)) : error("Not prob. vec!")
	WIFS(ifs::IFS, weights::Vector) = WIFS(ifs.linear, ifs.trans, weights)
	# WIFS(ifs::IFS, weights::Vector) = isprobvec(weights) ? new(ifs.linear, ifs.trans, weights, ifs.numMaps, ifs.dimAmbient) : error("Not prob. vec!")
	WIFS(ifs::IFS) = WIFS(ifs, ones(ifs.numMaps) ./ ifs.numMaps)

end

struct IFSNonlinear
	maps::Vector{Function}
	weights::Vector
	numMaps::Int
	dimAmbient::Int

	IFSNonlinear(maps::Vector{Function}, weights::Vector, dimAmbient::Int) = new(maps, weights, size(maps, 1), dimAmbient)
	IFSNonlinear(maps::Vector{Function}, dimAmbient::Int) = new(maps, ones(size(maps, 1)) ./ size(maps, 1), size(maps, 1), dimAmbient)
end

## functions
"""
Iterate points via probabilistic method.
"""
function itrPtsProb(linearIFS::Vector{Matrix{Float64}}, transIFS::Vector{Vector{Float64}},
	weights::Vector{Float64}, maxNumPts::Int, initPt)::Vector{Vector{Float64}}
	probDistr = Categorical(weights)
	ptsSet = fill(zeros(Float64, 2), maxNumPts)
	ptsSet[1] = initPt
	for i = 2:maxNumPts
		temptIndex = rand(probDistr)
		ptsSet[i] = linearIFS[temptIndex] * ptsSet[i - 1] + transIFS[temptIndex]
	end
	return ptsSet
end
getFixedPt(linear, trans) = ([1 0; 0 1] - linear[1]) \ trans[1]
itrPtsProb(linear, trans, weight, maxNumPts) = itrPtsProb(linear, trans, weight, maxNumPts, getFixedPt(linear, trans))
itrPtsProb(wifs::WIFS, maxNumPts::Int) = itrPtsProb(wifs.linear, wifs.trans, wifs.weights, maxNumPts)
itrPtsProb(wifs::WIFS, maxNumPts::Int, initPt) = itrPtsProb(wifs.linear, wifs.trans, wifs.weights, maxNumPts, initPt)
itrPtsProb(ifs::IFS, maxNumPts::Int) = itrPtsProb(WIFS(ifs), maxNumPts)
itrPtsProb(ifs::IFS, maxNumPts::Int, initPt) = itrPtsProb(WIFS(ifs), maxNumPts, initPt)
function itrPtsProb(ifs::IFSNonlinear, maxNumPts::Int=1000, initialPt::Vector{Float64}=[0., 0.])
	maps = ifs.maps
	probDistr = Categorical(ifs.weights)
	ptsSet = fill(zeros(Float64, 2), maxNumPts)
	ptsSet[1] = initialPt
	for i = 2:maxNumPts
		ptsSet[i] = maps[rand(probDistr)](ptsSet[i - 1])
	end
	return ptsSet
end

"add methods to determine the shape of IFS"
# size(wifs::WIFS) = size(wifs.trans, 1), size(wifs.trans[1], 1)
# size(ifs::IFS) = size(ifs.trans, 1), size(ifs.trans[1], 1)

## PredefinedIFS
"PredefinedIFS as a module"
module PredefinedIFS

import ..IFS
import ..WIFS

SierpinskiTriangle = IFS([[1 / 2 0; 0 1 / 2],
				  [1 / 2 0; 0 1 / 2],
				  [1 / 2 0; 0 1 / 2]],
				 [[0, 0],
				  [1 / 2, 0],
				  [1 / 4, 1 / 4 * sqrt(3)]])

BarnsleyFern = WIFS([[0  0; 0 0.16],
					 [ 0.85  0.04; -0.04 0.85],
					 [ 0.2  -0.26; 0.23 0.22],
					 [-0.15  0.28; 0.26 0.24]],
					[[0, 0],
					 [0, 1.6],
					 [0, 1.6],
				 	 [0, 0.44]],
					[0.01, 0.84, 0.08, 0.07])

HeighwayDragon = IFS([[1 / 2 -1 / 2; 1 / 2 1 / 2],
					[-1 / 2 -1 / 2; 1 / 2 -1 / 2]],
					[[0., 0.], [1.,0.]])

Twindragon = IFS([[1 / 2 -1 / 2; 1 / 2 1 / 2],
					[-1 / 2 1 / 2; -1 / 2 -1 / 2]],
					[[0., 0.], [1., 0.]])

Terdragon = IFS([[1 / 2 1 / (2 * sqrt(3)); -1 / (2 * sqrt(3)) 1 / 2],
					[0 -1 / sqrt(3); 1 / sqrt(3) 0],
					[1 / 2 1 / (2 * sqrt(3)); -1 / (2 * sqrt(3)) 1 / 2]],
					[[0, 0], [1 / 2, -1 / (2 * sqrt(3))], [1 / 2, 1 / (2 * sqrt(3))]])

function BaranskiCarpet(v::Vector=[0.3, 0.7], h::Vector=[0.1, 0.8, 0.1], pos::Matrix=[1 0 1; 0 1 1])::IFS
	pos = reverse(pos; dims=1)
	linear = [ [h[x[2]] 0; 0 v[x[1]]] for x in findall(pos .> 0)]
	trans = [ [sum(h[1:x[2] - 1]),  sum(v[1:x[1] - 1])] for x in findall(pos .> 0)]

	return IFS(linear, trans)
end

BedMcCarpet(pos::Matrix=[1 0 1; 0 1 1]) = BaranskiCarpet(ones(size(pos, 1)) ./ size(pos, 1), ones(size(pos, 2)) ./ size(pos, 2), pos)

# function BedMcCarpet(v::Int, h::Int, pos::Matrix)::IFS
# 	pos = reverse(pos; dims=1)
# 	numMaps = sum(pos)
# 	linear = fill([1.0/h 0.; 0 1.0/v], numMaps)
# 	trans = [ [(x[2]-1.0)/h, (x[1]-1)/v] for x in findall(pos .> 0)]

# 	return IFS(linear, trans)
# end

end # end of module PredefinedIFS
end # end of module IFS
