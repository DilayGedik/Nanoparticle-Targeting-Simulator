from simulation import TargetingConfig, simulate_targeting, final_summary
config = TargetingConfig(particle_count=300,receptor_density_relative=1.4)
result = simulate_targeting(config)
print(final_summary(result))
result["metrics"].to_csv("targeting_metrics.csv",index=False)
