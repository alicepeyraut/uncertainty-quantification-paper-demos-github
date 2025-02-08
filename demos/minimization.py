import compute_disp

import fire
import dolfin
import scipy.optimize
import numpy
import copy

import os
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)


def minimization(position = 'Supine', noise_level = 0.1, parameters_to_identify = {}, parameter_biased = None, iteration = None, dirpath = None, initialization = []):
    logging.getLogger('FFC').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    noise_level, iteration = float(noise_level), float(iteration) ### getting noise level to apply and iteration number for writing files (avoiding memory access problem by adding iteration number while writing)
    number_parameters = len(parameters_to_identify)
    if position == 'Supine+Prone':
        ### getting supine information
        ######### mesh information
        position = 'Supine'
        mesh = dolfin.Mesh()
        dolfin.XDMFFile(dirpath + "/supine/cubesupine.xdmf").read(mesh) ### retrieving the mesh already written
        dVmeas_supine = dolfin.Measure("dx", domain=mesh) ### defining measures on the mesh
        V_supine = dolfin.assemble(dolfin.Constant(1) * dVmeas_supine) ### volume mesh in supine
        ### displacement information
        new_directory = "ref" + str(position)
        res_basename_meas = os.path.join(dirpath, new_directory) ### defining the directory in which the displacement field were written
        U_mes_fs = dolfin.VectorFunctionSpace(
             mesh=mesh,
             family="Lagrange",
             degree=1) ### creating function space to retrieve displacement fields
        Umeas_supine = dolfin.Function(U_mes_fs, res_basename_meas+"/displacement_exhal_to_inhal.xml") ### displacement field unloading problem
        Umeas_norm_supine = (dolfin.assemble(dolfin.inner(Umeas_supine, Umeas_supine) * dVmeas_supine)/2/V_supine)**(1/2)
        ### getting prone information
        ######### mesh information
        position = 'Prone'
        mesh = dolfin.Mesh()
        dolfin.XDMFFile(dirpath + "/prone/cubeprone.xdmf").read(mesh) ### retrieving the mesh already written
        dVmeas_prone = dolfin.Measure("dx", domain=mesh)
        V_prone = dolfin.assemble(dolfin.Constant(1) * dVmeas_prone)
        ### displacement information
        new_directory = "ref" + str(position)
        res_basename_meas = os.path.join(dirpath, new_directory) ### defining the directory in which the displacement field were written
        U_mes_fs = dolfin.VectorFunctionSpace(
             mesh=mesh,
             family="Lagrange",
             degree=1) ### creating function space to retrieve displacement fields
        Umeas_prone = dolfin.Function(U_mes_fs, res_basename_meas+"/displacement_exhal_to_inhal.xml") ## displacement field unloading problem
        Umeas_norm_prone = (dolfin.assemble(dolfin.inner(Umeas_prone, Umeas_prone) * dVmeas_prone)/2/V_prone)**(1/2)
        ### adding noise to the displacement field
        ###### in supine position
        scale_supine = noise_level * Umeas_norm_supine ### scale depends on the noise level
        noise_supine = Umeas_supine.copy(deepcopy=True)
        noise_supine.vector()[:] = numpy.random.normal(loc=0.0, scale=scale_supine, size=Umeas_supine.vector().get_local().shape) ### creating random noise with null mean and standard deviation scale_supine, of the size of U_meas_supine
        noise_norm_supine = (dolfin.assemble(dolfin.inner(noise_supine, noise_supine) * dVmeas_supine)/2/V_supine)**(1/2)
        Umeas_supine.vector()[:] += noise_supine.vector()[:] ### adding noise to displacement field
        Umeas_supine_noise_norm = (dolfin.assemble(dolfin.inner(Umeas_supine, Umeas_supine) * dVmeas_supine)/2/V_supine)**(1/2) ### norm of the noisy displacement field
        ###### in prone position
        scale_prone = noise_level * Umeas_norm_prone
        noise_prone = Umeas_prone.copy(deepcopy=True)
        noise_prone.vector()[:] = numpy.random.normal(loc=0.0, scale=scale_prone, size=Umeas_prone.vector().get_local().shape) ### creating random noise with null mean and standard deviation scale_prone, of the size of U_meas_prone
        noise_norm_prone = (dolfin.assemble(dolfin.inner(noise_prone,noise_prone) * dVmeas_prone)/2/V_prone)**(1/2)
        Umeas_prone.vector()[:] += noise_prone.vector()[:] ### adding noise to displacement field
        Umeas_prone_noise_norm = (dolfin.assemble(dolfin.inner(Umeas_prone, Umeas_prone) * dVmeas_prone)/2/V_prone)**(1/2)
        ### starting minimization
        sol = scipy.optimize.minimize(L_prone_and_supine, initialization, args=(Umeas_supine, Umeas_prone, noise_norm_supine, Umeas_supine_noise_norm, noise_norm_prone, Umeas_prone_noise_norm, noise_level, iteration, parameters_to_identify, dirpath, parameter_biased, V_prone, V_supine), options={'maxiter': 150, 'maxfev': 150, 'xatol': 5e-2}, method="Nelder-Mead")
    else:
        ### mesh information
        mesh = dolfin.Mesh()
        new_directory = "ref" + str(position)
        if position == "Supine":
            name_mesh=dirpath+"/supine/cubesupine.xdmf"
        else:
            name_mesh=dirpath+"/prone/cubeprone.xdmf" ### retrieving the right displacement field whether the patient is in prone or in supine position
        res_basename_meas = os.path.join(dirpath, new_directory)
        dolfin.XDMFFile(name_mesh).read(mesh)
        dVmeas = dolfin.Measure("dx", domain=mesh)
        V0 = dolfin.assemble(dolfin.Constant(1) * dVmeas)
        ### retrieving displacement field
        U_mes_fs = dolfin.VectorFunctionSpace(
             mesh=mesh,
             family="Lagrange",
             degree=1) ### function space
        Umeas = dolfin.Function(U_mes_fs, res_basename_meas+"/displacement_exhal_to_inhal.xml") ### displacement field from end-exhalation to unloaded configuration
        Umeas_norm = (dolfin.assemble(dolfin.inner(Umeas, Umeas)*dVmeas)/2/V0)**(1/2)
        ### scale 
        scale = noise_level * Umeas_norm
        noise = Umeas.copy(deepcopy=True)
        noise.vector()[:] = numpy.random.normal(loc = 0.0, scale = scale, size = Umeas.vector().get_local().shape)
        noise_norm = (dolfin.assemble(dolfin.inner(noise,noise)*dVmeas)/2/V0)**(1/2)
        Umeas.vector()[:] += noise.vector()[:] ### adding noise to the displacement field
        Umeas_noise_norm = (dolfin.assemble(dolfin.inner(Umeas, Umeas) * dVmeas)/2/V0)**(1/2)
        # print("Umeas_noise_norm", Umeas_noise_norm)
        sol = scipy.optimize.minimize(L_prone_or_supine, initialization, args=(Umeas, V0, noise_norm, Umeas_noise_norm, noise_level, iteration, position, parameters_to_identify, dirpath, parameter_biased), options={'maxiter': 150, 'maxfev': 150,'xatol': 5e-2}, method="Nelder-Mead")
    if sol.success == True:
        for param in initialization:
            print(param)
        for k in range(0, number_parameters):
            print(sol.x[k])
        print(sol.fun)
    return



def L_prone_or_supine(x, Umeas, V0, noise_norm, Umeas_noise_norm, noise_level, iteration, position, parameters_to_identify, dirpath, parameter_biased):
    j = 0
    parameters_to_identify_updated = {}
    for key, value in parameters_to_identify.items(): ### changing the value of the parameters to identify at each iteration
        if key == "alpha":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_1":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_2":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_3":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_healthy":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "gamma":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "c1":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "c2":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "pe":
            parameters_to_identify_updated[key] = - numpy.log(numpy.exp(abs(x[j]))) ### ensuring negativity of the parameter
            j += 1
        elif key == "pi":
            parameters_to_identify_updated[key] = - numpy.log(numpy.exp(abs(x[j]))) ### ensuring negativity of the parameter
            j +=1
    if parameter_biased != None:
        for key, value in parameter_biased.items():
            parameters_to_identify_updated[key] = value
    try:
        if noise_norm != 0: ### for normalization
            normalization = noise_norm
        else:
            normalization = Umeas_noise_norm
        # print("normalization", normalization)
        U, dV = compute_disp.compute_disp(position = position, parameters_to_identify = parameters_to_identify_updated, noise = noise_level, dirpath = dirpath, iteration = iteration) ### compute the displacement field from end-exhalation to end-inhalation with the parameter values of the current iteration
        U_err = U.copy(deepcopy = True)
        U_err.vector()[:] -= Umeas.vector()[:]
        L = (dolfin.assemble(dolfin.inner(U_err,U_err) * dV)/2/V0)**(1/2)/normalization ### cost function
    except:
        L = 1e20*abs(x[0]) ### if computation did not converge
    return(L)



def L_prone_and_supine(x, Umeas_supine, Umeas_prone, noise_norm_supine, Umeas_supine_noise_norm, noise_norm_prone, Umeas_prone_noise_norm, noise_level, iteration, parameters_to_identify, dirpath, parameter_biased, V_prone, V_supine):
    j = 0
    parameters_to_identify_updated = {}
    for key, value in parameters_to_identify.items(): ### changing the value of the parameters to identify at each iteration
        if key == "alpha":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_1":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_2":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_fibrose_3":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "alpha_healthy":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "gamma":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "c1":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "c2":
            parameters_to_identify_updated[key] = numpy.log(numpy.exp(x[j])) ### ensuring positivity of the parameter
            j += 1
        elif key == "pe":
            parameters_to_identify_updated[key] = - numpy.log(numpy.exp(abs(x[j]))) ### ensuring negativity of the parameter
            j += 1
        elif key == "pi":
            parameters_to_identify_updated[key] = - numpy.log(numpy.exp(abs(x[j]))) ### ensuring negativity of the parameter
            j +=1
    if parameter_biased != None:
        for key, value in parameter_biased.items():
            parameters_to_identify_updated[key] = value
    try:
        if noise_norm_supine != 0 and noise_norm_prone != 0: ### for normalization
            normalization_supine = noise_norm_supine
            normalization_prone = noise_norm_prone
        else:
            normalization_supine = Umeas_supine_noise_norm
            normalization_prone = Umeas_prone_noise_norm
        ### computing displacement field in supine position
        U_supine, dV_supine = compute_disp.compute_disp(position = 'Supine', parameters_to_identify = parameters_to_identify_updated, noise = noise_level, dirpath = dirpath, iteration = iteration) ### compute the displacement field from end-exhalation to end-inhalation with the parameter values of the current iteration
        U_err_supine = U_supine.copy(deepcopy = True)
        U_err_supine.vector()[:] -= Umeas_supine.vector()[:]
        L = (dolfin.assemble(dolfin.inner(U_err_supine,U_err_supine) * dV_supine)/2/V_supine)**(1/2)/normalization_supine ### cost function
        ### computing displacement field in prone position
        U_prone, dV_prone = compute_disp.compute_disp(position = 'Prone', parameters_to_identify = parameters_to_identify_updated, noise = noise_level, dirpath = dirpath, iteration = iteration) ### compute the displacement field from end-exhalation to end-inhalation with the parameter values of the current iteration
        U_err_prone = U_prone.copy(deepcopy = True)
        U_err_prone.vector()[:] -= Umeas_prone.vector()[:]
        L += (dolfin.assemble(dolfin.inner(U_err_prone, U_err_prone) * dV_prone)/2/V_prone)**(1/2)/normalization_prone ### adding second term to cost function
    except:
        L = 1e20*abs(x[0]) ### if computation did not converge
    return(L)

if (__name__ == "__main__"):
    fire.Fire(minimization)
