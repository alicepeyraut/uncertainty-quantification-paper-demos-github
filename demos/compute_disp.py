#coding=utf8

################################################################################
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_mech     as dmech

import fire
import shutil
import os

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

################################################################################

def compute_disp(position = None, parameters_to_identify = {}, noise = None, dirpath = None, iteration = None) :

    logging.getLogger('FFC').setLevel(logging.WARNING)

####################################################################### parameters values definition ###
    alpha, alpha_healthy, alpha_fibrose, alpha_fibrose_1, alpha_fibrose_2, alpha_fibrose_3, gamma, c1, c2, pe, pi = 0.16, 0.16, 0.67, 0.67, 0.9, 1.6, 0.5, 0.2, 0.4, -0.5, -2 ### reference parameter values, stiffnesses alpha in kPa, gamma [-], c1 and c2 in kPa, and the pressures pe and pi in kPa as well
    res_basename = "params"
    
    for key, value in parameters_to_identify.items(): ### retrieving value of parameters to identify
        if key == "alpha":
            alpha = float(value)
            number_zones = 1 ### if alpha, one zone
        elif key == "alpha_healthy":
            alpha_healthy = float(value)
            number_zones = 2 ### if alpha_healthy, at least two zones
        elif key == "alpha_fibrose":
            alpha_fibrose = float(value)
            number_zones = 2 ### if alpha_fibrose, exactly two zones
        elif key == "alpha_fibrose_1":
            alpha_fibrose_1 = float(value)
            number_zones = 3 ### if alpha_fibrose_1, at least three zones
        elif key == "alpha_fibrose_2":
            alpha_fibrose_2 = float(value)
            number_zones = 3 ### if alpha_fibrose_1, at least three zones
        elif key == "alpha_fibrose_3":
            alpha_fibrose_3 = float(value)
            number_zones = 4 ### if alpha_fibrose_1, at least four zones (more zones are not implemented)
        elif key == "gamma":
            gamma = float(value)
        elif key == "c1":
            c1 = float(value)
        elif key == "c2":
            c2 = float(value)
        elif key == "pe":
            pe = float(value)
        elif key == "pi":
            pi = float(value)
        res_basename += key
        res_basename += str(value)[:7]
    res_basename += "iteration"
    res_basename += str(iteration)
    res_basename += "noise"
    res_basename += str(noise) ### directory at which the computation files are written

    ### material definition
    if number_zones==2: ### defining the parameter values for each zone
        params_healthy = {
            "alpha": alpha_healthy,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5}
        params_fibrose = {
            "alpha": alpha_fibrose,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5}
        mat_params_healthy = {"scaling":"linear", "parameters": params_healthy}
        mat_params_fibrose = {"scaling":"linear", "parameters": params_fibrose}
        mat_params = [mat_params_fibrose, mat_params_healthy] 
    elif number_zones==3: ### defining the parameter values for each zone
        params_healthy = {
            "alpha": alpha_healthy,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        params_fibrose_1 = {
            "alpha": alpha_fibrose_1,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        params_fibrose_2 = {
            "alpha": alpha_fibrose_2,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        mat_params_healthy = {"scaling":"linear", "parameters": params_healthy}
        mat_params_fibrose_1 = {"scaling":"linear", "parameters": params_fibrose_1}
        mat_params_fibrose_2 = {"scaling":"linear", "parameters": params_fibrose_2}
        mat_params = [mat_params_fibrose_1, mat_params_fibrose_2, mat_params_healthy]
    elif number_zones==4: ### defining the parameter values for each zone
        params_healthy = {
            "alpha": alpha_healthy,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        params_fibrose_1 = {
            "alpha": alpha_fibrose_1,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        params_fibrose_2 = {
            "alpha": alpha_fibrose_2,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        params_fibrose_3 = {
            "alpha": alpha_fibrose_3,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5,
            "rho_solid":1.06e-6}
        mat_params_healthy = {"scaling":"linear", "parameters": params_healthy}
        mat_params_fibrose_1 = {"scaling":"linear", "parameters": params_fibrose_1}
        mat_params_fibrose_2 = {"scaling":"linear", "parameters": params_fibrose_2}
        mat_params_fibrose_3 = {"scaling":"linear", "parameters": params_fibrose_3}
        mat_params = [mat_params_fibrose_1, mat_params_fibrose_2, mat_params_fibrose_3, mat_params_healthy]
    else:
        params = {
            "alpha": alpha,
            "gamma":gamma,
            "c1":c1,
            "c2":c2,
            "kappa":1e2,
            "eta":1e-5}
        mat_params = {"scaling":"linear", "parameters": params}
        mat_params = mat_params
    

####################################################################### defining directories and parameters depending on the position ###
    if noise == '':
        res_basename = "ref" + str(position)

    new_directory = os.path.join(dirpath, res_basename)
    if str(position) == "Prone": ### defining parameters depending on the position
        coef = -1.
        cube_params = {"path_and_mesh_name": dirpath + "/prone/cubeprone.xdmf"}
        porosity_params_unloading={"type": "from_file", "val": dirpath + "/prone/prone-poro.xml"}
    elif str(position) == "Supine":
        coef = +1.
        cube_params = {"path_and_mesh_name": dirpath + "/supine/cubesupine.xdmf"}
        porosity_params_unloading={"type": "from_file", "val": dirpath + "/supine/supine-poro.xml"}
    else:
        print("Warning, there is a problem: gravity should be Prone or Supine... aborting...")

    try:
        shutil.rmtree(new_directory)
    except OSError:
        pass
    os.mkdir(new_directory)

    ####################################################################### defining loading ###
    load_params_inverse = {
        "type":"p_boundary_condition0", "f":coef*9.81e3, "P0" : float(pe)}
    load_params_direct = {
        "type":"p_boundary_condition", "f":coef*9.81e3, "P0" : float(pi)}

    ####################################################################### computing displacement field from end-exhalation to end-inhalation ###
    try:
        U_unloading, phis_unloading, dV_unloading = dmech.run_RivlinCube_PoroHyperelasticity( ### unloading problem
            dim=3,
            inverse=1,
            cube_params=cube_params,
            porosity_params=porosity_params_unloading,
            get_results = 1,
            mat_params=mat_params,
            step_params={"dt_min":1e-4},
            load_params=load_params_inverse,
            res_basename=new_directory+"/unloading",
            inertia_params={"applied":True},
            plot_curves=0,
            verbose=1)

        U_loading, phis_loading, dV_loading = dmech.run_RivlinCube_PoroHyperelasticity( ### loading problem
            dim=3,
            inverse=0,
            porosity_params= {"type": "function_xml_from_array", "val": phis_unloading},
            cube_params=cube_params,
            mat_params=mat_params,
            step_params={"dt_ini":0.125, "dt_min":1e-4},
            load_params=load_params_direct,
            res_basename = new_directory+"/loading",
            move_params = {"move":True, "U": U_unloading},
            inertia_params={"applied":True},
            get_results = 1,
            plot_curves=0,
            verbose=1)

        U_exhal_to_inhal = U_unloading.copy(deepcopy =True)
        U_exhal_to_inhal.vector()[:] += U_loading.vector().get_local()[:]

        # print(noise, new_directory )
        if noise != '':
            shutil.rmtree(new_directory)
        return(U_exhal_to_inhal, dV_unloading)
    except:
        shutil.rmtree(new_directory)
        return



if (__name__ == "__main__"):
    fire.Fire(compute_disp)
