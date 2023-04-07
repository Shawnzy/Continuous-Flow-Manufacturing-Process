import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df.info(verbose=True)


# Grouping columns by categories and machines
def group_columns_by_property(property_str):
    return [f"Machine{i}.{property_str}" for i in range(1, 4)]


raw_material_columns = [
    group_columns_by_property("RawMaterial.Property" + str(i)) for i in range(1, 5)
]
feeder_parameter_columns = group_columns_by_property(
    "RawMaterialFeederParameter.U.Actual"
)
zone_temperature_columns = [
    group_columns_by_property(f"Zone{i}Temperature.C.Actual") for i in range(1, 3)
]
motor_amperage_columns = group_columns_by_property("MotorAmperage.U.Actual")
motor_rpm_columns = group_columns_by_property("MotorRPM.C.Actual")
material_pressure_columns = group_columns_by_property("MaterialPressure.U.Actual")
material_temperature_columns = group_columns_by_property("MaterialTemperature.U.Actual")
exit_temperature_columns = group_columns_by_property("ExitZoneTemperature.C.Actual")


# Function to plot columns
def plot_columns(df, columns, title):
    plt.figure(figsize=(15, 5))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


# Plotting grouped columns
for i, group in enumerate(raw_material_columns, start=1):
    plot_columns(df, group, f"Raw Material Property {i}")
plot_columns(df, feeder_parameter_columns, "Raw Material Feeder Parameters")
for i, group in enumerate(zone_temperature_columns, start=1):
    plot_columns(df, group, f"Zone {i} Temperatures")
plot_columns(df, motor_amperage_columns, "Motor Amperage")
plot_columns(df, motor_rpm_columns, "Motor RPM")
plot_columns(df, material_pressure_columns, "Material Pressure")
plot_columns(df, material_temperature_columns, "Material Temperature")
plot_columns(df, exit_temperature_columns, "Exit Zone Temperatures")

#######################################

stage_output_columns = [f"Stage1.Output.Measurement{i}.U.Actual" for i in range(15)]


# Function to plot individual column
def plot_individual_column(df, column):
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[column], label=column)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(column)
    plt.legend(loc="upper right")
    plt.show()


# Plotting individual stage output columns
for col in stage_output_columns:
    plot_individual_column(df, col)
