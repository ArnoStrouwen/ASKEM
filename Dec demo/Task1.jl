using Pkg; Pkg.activate("."); Pkg.instantiate()
using CSV
using DataFrames
using Turing
using OrdinaryDiffEq
using SciMLExpectations, IntegralsCubature
using Optimization, NLopt
using Plots
using StatsPlots

# dataset covid hospitalizations in NY
# https://health.data.ny.gov/Health/New-York-State-Statewide-COVID-19-Hospitalizations/jw46-jpb7/explore/query/SELECT%0A%20%20%60as_of_date%60%2C%0A%20%20sum%28%60patients_currently%60%29%20AS%20%60sum_patients_currently%60%0AWHERE%0A%20%20%28%60ny_forward_region%60%20IN%20%28%27NEW%20YORK%20CITY%27%29%29%0A%20%20AND%20%28%60as_of_date%60%0A%20%20%20%20%20%20%20%20%20BETWEEN%20%272020-12-01T00%3A00%3A00%27%20%3A%3A%20floating_timestamp%0A%20%20%20%20%20%20%20%20%20AND%20%272021-04-01T00%3A00%3A00%27%20%3A%3A%20floating_timestamp%29%0AGROUP%20BY%20%60as_of_date%60/page/column_manager
data = CSV.read("dataset.csv", DataFrame)
hospitalized = parse.(Int,replace.(data[!,:Hospitalized],',' => ""))
hospitalized = hospitalized[1:end-21]
days = 1:length(hospitalized)
tspan = (days[begin],days[end])
function SIR(du, u, p, t)
    # Model parameters.
    β, γ = p
    # Current state.
    S, I, R = u
    # Evaluate differential equations.
    du[1] = dS = - β*S*I
    du[2] = dI = β*S*I - γ*I
    du[3] = dR = γ*I
    return nothing
end
# assumed parameters from literature
hospitalization_rate_guess = 0.05
S0_guess = 1_400_000.0
I0_guess  = hospitalized[1]/hospitalization_rate_guess
R0_guess = 0.0
u0 = [S0_guess,I0_guess,R0_guess]
γ_guess = 1/9
β_guess = (2^(1/14)-1 + γ_guess)/S0_guess

p = [β_guess,γ_guess]
prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
sol = solve(prob,Tsit5())
plot(days, sol[2,:]*hospitalization_rate_guess,label="fit");
scatter!(days,hospitalized,label="data");
plot!(;xlabel="day",ylabel="hospitalizations",title="Literature Model")
savefig("fig1")
# improve on parameters by probabilistic programming
@model function fitCHIME(hospitalized, prob)
    # Prior distributions informed by literature.
    σ ~ Uniform(0.0,1000.0)

    hospitalization_rate ~ Uniform(0.01,0.1)

    βmillion ~ Uniform(0.05,0.25)
    γ ~ Uniform(0.05,0.25)

    S0 ~ Uniform(0.5e6,2e6)
    I0 ~ Uniform(0.0,100_000.0)
    R0 = 0.0
    p = [βmillion/1e6, γ]
    u0 = [S0,I0,R0]
    # Simulate the SIR model
    predicted = solve(prob, Tsit5();u0=u0, p=p)
    failure = size(predicted, 2) < length(prob.kwargs.data.saveat)
    if failure
        println("failure")
        Turing.DynamicPPL.acclogp!!(__varinfo__, -Inf)
        return
    end

    # Observations.
    for i in eachindex(predicted)
        hospitalized[i] ~ Normal(predicted[2,i]*hospitalization_rate, σ)
    end
    return nothing
end
model = fitCHIME(hospitalized, prob)

#chain = sample(model, NUTS(0.65), MCMCThreads(), 1_000, 4)
#write("chain-file.jls", chain)
chain = read("chain-file.jls", Chains)

βdata = chain[:βmillion]./1_000_000
γdata = chain[:γ]
S0data = chain[:S0]
I0data = chain[:I0]
hospitalization_ratedata = chain[:hospitalization_rate]

plot(;legend=false,xlabel="day",ylabel="hospitalizations",title="PPL model");
for i in eachindex(βdata)
    β = βdata[i]
    γ = γdata[i]
    p = [β,γ]
    S0 = S0data[i]
    I0 = I0data[i]
    u0 = [S0,I0,0.0]
    hospitalization_rate = hospitalization_ratedata[i]
    prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
    sol = solve(prob,Tsit5())
    plot!(days, sol[2,:]*hospitalization_rate);
end
scatter!(days,hospitalized)
savefig("fig2")

# go over some interventions
plot(;legend=false, title="β reduced by 15%",xlabel="day",ylabel="hospitalizations");
for i in eachindex(βdata)
    β = βdata[i]*0.85
    γ = γdata[i]
    p = [β,γ]
    S0 = S0data[i]
    I0 = I0data[i]
    u0 = [S0,I0,0.0]
    hospitalization_rate = hospitalization_ratedata[i]
    prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
    sol = solve(prob,Tsit5())
    plot!(days, sol[2,:]*hospitalization_rate);
end
scatter!(days,hospitalized);
hline!([2500.0],c=:red,ms=10)
savefig("fig3")

plot(;legend=false, title="Closing primary schools reduces β by 10-20%",xlabel="day",ylabel="hospitalizations");
for i in eachindex(βdata)
    β = βdata[i]*rand(Truncated(Normal(0.85,0.025),0.8,0.9))
    γ = γdata[i]
    p = [β,γ]
    S0 = S0data[i]
    I0 = I0data[i]
    u0 = [S0,I0,0.0]
    hospitalization_rate = hospitalization_ratedata[i]
    prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
    sol = solve(prob,Tsit5())
    plot!(days, sol[2,:]*hospitalization_rate);
end
scatter!(days,hospitalized);
hline!([2500.0],c=:red,ms=10)
savefig("fig4")

plot(;legend=false, title="Closing secondary schools reduces β by 5-15%",xlabel="day",ylabel="hospitalizations");
for i in eachindex(βdata)
    β = βdata[i]*rand(Truncated(Normal(0.9,0.025),0.85,0.95))
    γ = γdata[i]
    p = [β,γ]
    S0 = S0data[i]
    I0 = I0data[i]
    u0 = [S0,I0,0.0]
    hospitalization_rate = hospitalization_ratedata[i]
    prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
    sol = solve(prob,Tsit5())
    plot!(days, sol[2,:]*hospitalization_rate);
end
scatter!(days,hospitalized);
hline!([2500.0],c=:red,ms=10)

plot(;legend=false, title="Measures after Christmas reducing β by 20%",xlabel="day",ylabel="hospitalizations");
for i in eachindex(βdata)
    β = βdata[i]
    γ = γdata[i]
    p = [β,γ]
    S0 = S0data[i]
    I0 = I0data[i]
    u0 = [S0,I0,0.0]
    hospitalization_rate = hospitalization_ratedata[i]
    prob = ODEProblem(SIR,u0,tspan,p,saveat=days)
    condition(u,t,integrator) = t==26
    affect!(integrator) = integrator.p[1] *= 0.8
    cb = DiscreteCallback(condition,affect!,save_positions=(false,false))
    sol = solve(prob,Tsit5(),callback=cb,tstops=[26])
    plot!(days, sol[2,:]*hospitalization_rate);
end
scatter!(days,hospitalized);
hline!([2500.0],c=:red,ms=10)
savefig("fig5")
