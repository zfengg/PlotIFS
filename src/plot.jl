# plot
using Plots

# functions
function plotPts(ptsSet::Vector{Vector{Float64}}, mc::String="#009AFA")
	xs = [x[1] for x in ptsSet]
	ys = [x[2] for x in ptsSet]
	scatter(xs, ys, 
			leg=false,
			markershape=:circle,
			markeralpha=0.7;
			markersize=1,
			markercolor=mc,
			# markerstroke=false,
    		markerstrokewidth = 0,
    		# markerstrokealpha = 0.2,
			# grid=false,
			showaxis=false,
			ticks=false,
			xlims=extrema(xs),
			ylims=extrema(ys),
			size=(680, 500))
end



