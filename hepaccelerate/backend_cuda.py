#need to set these CUDA variables explicitly
from numba import cuda

import cupy
import math
import numpy as np

@cuda.jit(device=True)
def searchsorted_devfunc(arr, val):
    ret = -1

    #overflow
    if val > arr[-1]:
        return len(arr)

    #underflow bin will not be filled
    if val < arr[0]:
        return -1

    for i in range(len(arr)):
        if val < arr[i+1]:
            ret = i
            break
    return ret

@cuda.jit(device=True)
def searchsorted_devfunc2D(arr_x, arr_y, val_x, val_y):
    ret = -1
    for i in range(len(arr_x)):
        for j in range(len(arr_y)):
            #if val <= arr[i]:
            if val_x < arr_x[i+1] and val_y <arr_y[i+1]:
                ret_i = i
                ret_j = j
                break
    return ret_i, ret_j

@cuda.jit
def searchsorted_kernel(vals, arr, inds_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(vals), xstride):
        inds_out[i] = searchsorted_devfunc(arr, vals[i])

def searchsorted(arr, vals):
    """
    Find indices to insert vals into arr to preserve order.
    """
    ret = cupy.zeros_like(vals, dtype=cupy.int32)
    searchsorted_kernel[32, 1024](vals, arr, ret)
    return ret 

@cuda.jit
def fill_histogram(data, weights, bins, out_w, out_w2):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(data), xstride):
        bin_idx = searchsorted_devfunc(bins, data[i])
        if bin_idx >=0 and bin_idx < len(out_w):
            cuda.atomic.add(out_w, bin_idx, weights[i])
            cuda.atomic.add(out_w2, bin_idx, weights[i]**2)

@cuda.jit
def select_opposite_sign_muons_cudakernel(muon_charges_content, muon_charges_offsets, content_mask_in, content_mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, muon_charges_offsets.shape[0]-1, xstride):
        start = muon_charges_offsets[iev]
        end = muon_charges_offsets[iev + 1]
        
        ch1 = 0
        idx1 = -1
        ch2 = 0
        idx2 = -1
        
        for imuon in range(start, end):
            if not content_mask_in[imuon]:
                continue
                
            if idx1 == -1:
                ch1 = muon_charges_content[imuon]
                idx1 = imuon
                continue
            else:
                ch2 = muon_charges_content[imuon]
                if (ch2 != ch1):
                    idx2 = imuon
                    content_mask_out[idx1] = 1
                    content_mask_out[idx2] = 1
                    break
    return

@cuda.jit
def sum_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] += content[ielem]

@cuda.jit
def multiply_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue

        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] *= content[ielem]
            
@cuda.jit
def max_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] > accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum

@cuda.jit
def calc_px_cudakernel(content_pt, content_phi, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iobj in range(xi, content_pt.shape[0]-1, xstride):
        out[iobj] = content_pt[iobj] * math.cos(content_phi[iobj])

@cuda.jit
def calc_py_cudakernel(content_pt, content_phi, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iobj in range(xi, content_pt.shape[0]-1, xstride):
        out[iobj] = content_pt[iobj] * math.sin(content_phi[iobj])

@cuda.jit
def calc_pz_cudakernel(content_pt, content_eta, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iobj in range(xi, content_pt.shape[0]-1, xstride):
        out[iobj] = content_pt[iobj] * math.sinh(content_eta[iobj])

@cuda.jit
def calc_en_cudakernel(content_pt, content_eta, content_mass, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iobj in range(xi, content_pt.shape[0]-1, xstride):
        out[iobj] = math.sqrt(content_mass[iobj]**2 + (1+math.sinh(content_eta[iobj])**2)*content_pt[iobj]**2)

@cuda.jit
def dnn_jets_cudakernel(content, offsets, feats_indx, nobj, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
        start = offsets[iev]
        end = offsets[iev + 1]

        for idx in range(nobj):
            index_to_get = 0
            for ielem in range(start, end):
                if mask_content[ielem]:
                    if index_to_get == idx:
                        out[iev][idx][feats_indx] = content[ielem]
                        break
                    else:
                        index_to_get += 1


@cuda.jit
def dnn_leps_cudakernel(content, feats_indx, mask_rows, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, content.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue

        out[iev][0][feats_indx] = content[iev]


@cuda.jit
def dnn_met_cudakernel(content, feats_indx, mask_rows, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, content.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue

        out[iev][feats_indx] = content[iev]

@cuda.jit
def min_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] < accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum
    
@cuda.jit
def get_in_offsets_cudakernel(content, offsets, indices, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
        start = offsets[iev]
        end = offsets[iev + 1]
        
        index_to_get = 0
        for ielem in range(start, end):
            if mask_content[ielem]:
                if index_to_get == indices[iev]:
                    out[iev] = content[ielem]
                    break
                else:
                    index_to_get += 1
        
@cuda.jit
def min_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] < accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum
        
def sum_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    sum_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=dtype)
    sum_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, sum_offsets)
    cuda.synchronize()
    return sum_offsets

def multiply_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    product_offsets = cupy.ones(len(struct.offsets) - 1, dtype=dtype)
    multiply_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, product_offsets)
    return product_offsets

def max_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    max_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def min_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    min_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def select_muons_opposite_sign(muons, in_mask):
    out_mask = cupy.invert(muons.make_mask())
    select_opposite_sign_muons_cudakernel[32,1024](muons.charge, muons.offsets, in_mask, out_mask)
    cuda.synchronize()
    return out_mask

def get_in_offsets(content, offsets, indices, mask_rows, mask_content):
    #out = cupy.zeros(len(offsets) - 1, dtype=content.dtype)
    out = -999.*cupy.ones(len(offsets) - 1, dtype=content.dtype) #to avoid histos being filled with 0 for non-existing objects, i.e. in events with no fat jets
    get_in_offsets_cudakernel[32, 1024](content, offsets, indices, mask_rows, mask_content, out)
    cuda.synchronize()
    return out

"""
For all events (N), mask the objects in the first collection (M1) if they are closer than dr2 to any object in the second collection (M2).

    etas1: etas of the first object, array of (M1, )
    phis1: phis of the first object, array of (M1, )
    mask1: mask (enabled) of the first object, array of (M1, )
    offsets1: offsets of the first object, array of (N, )

    etas2: etas of the second object, array of (M2, )
    phis2: phis of the second object, array of (M2, )
    mask2: mask (enabled) of the second object, array of (M2, )
    offsets2: offsets of the second object, array of (N, )
    
    mask_out: output mask, array of (M1, )

"""
@cuda.jit
def mask_deltar_first_cudakernel(etas1, phis1, mask1, offsets1, etas2, phis2, mask2, offsets2, dr2, mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(offsets1)-1, xstride):
        a1 = offsets1[iev]
        b1 = offsets1[iev+1]
        
        a2 = offsets2[iev]
        b2 = offsets2[iev+1]
        
        for idx1 in range(a1, b1):
            if not mask1[idx1]:
                continue
                
            eta1 = etas1[idx1]
            phi1 = phis1[idx1]
            for idx2 in range(a2, b2):
                if not mask2[idx2]:
                    continue
                eta2 = etas2[idx2]
                phi2 = phis2[idx2]
                
                deta = abs(eta1 - eta2)
                dphi = (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi
                
                #if first object is closer than dr2, mask element will be *disabled*
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr
                
def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1.eta.shape)
    assert(mask2.shape == objs2.eta.shape)
    assert(objs1.offsets.shape == objs2.offsets.shape)
    
    mask_out = cupy.zeros_like(objs1.eta, dtype=cupy.bool)
    mask_deltar_first_cudakernel[32, 1024](
        objs1.eta, objs1.phi, mask1, objs1.offsets,
        objs2.eta, objs2.phi, mask2, objs2.offsets,
        drcut**2, mask_out
    )
    cuda.synchronize()
    mask_out = cupy.invert(mask_out)
    return mask_out

@cuda.jit
def mask_overlappingAK4_cudakernel(etas1, phis1, mask1, offsets1, etas2, phis2, mask2, offsets2, tau32, tau21, dr2, tau32cut, tau21cut, mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(offsets1)-1, xstride):
        a1 = offsets1[iev]
        b1 = offsets1[iev+1]
        
        a2 = offsets2[iev]
        b2 = offsets2[iev+1]
        
        for idx1 in range(a1, b1):
            if not mask1[idx1]:
                continue
                
            eta1 = etas1[idx1]
            phi1 = phis1[idx1]
            for idx2 in range(a2, b2):
                if not mask2[idx2]:
                    continue
                eta2 = etas2[idx2]
                phi2 = phis2[idx2]
                
                deta = abs(eta1 - eta2)
                dphi = (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi
                
                #if first object is closer than dr2, mask element will be *disabled*
                passdr = ((deta**2 + dphi**2) < dr2)
                if passdr:
                  passtau32 = (tau32[idx2] < tau32cut)
                  passtau21 = (tau21[idx2] < tau21cut)
                  mask_out[idx1] = (passtau32 or passtau21)

def mask_overlappingAK4(objs1, mask1, objs2, mask2, drcut, tau32cut, tau21cut):
    assert(mask1.shape == objs1.eta.shape)
    assert(mask2.shape == objs2.eta.shape)
    assert(objs1.offsets.shape == objs2.offsets.shape)
    
    mask_out = cupy.zeros_like(objs1.eta, dtype=cupy.bool)
    mask_overlappingAK4_cudakernel[32, 1024](
        objs1.eta, objs1.phi, mask1, objs1.offsets,
        objs2.eta, objs2.phi, mask2, objs2.offsets, objs2.tau32, objs2.tau21,
        drcut**2, tau32cut, tau21cut, mask_out
    )
    cuda.synchronize()
    mask_out = cupy.invert(mask_out)
    return mask_out

def histogram_from_vector(data, weights, bins):        
    out_w = cupy.zeros(len(bins) - 1, dtype=cupy.float32)
    out_w2 = cupy.zeros(len(bins) - 1, dtype=cupy.float32)
    if not data.shape[0] == 0:
        fill_histogram[32, 1024](data, weights, bins, out_w, out_w2)
    return cupy.asnumpy(out_w), cupy.asnumpy(out_w2), cupy.asnumpy(bins)


@cuda.jit
def get_bin_contents_cudakernel(values, edges, contents, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for i in range(xi, len(values), xstride):
        v = values[i]
        ibin = searchsorted_devfunc(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

@cuda.jit
def get_bin_contents2D_cudakernel(values_x, values_y, edges_x, edges_y, contents, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for i in range(xi, len(values_x), xstride):
        v_x = values_x[i]
        v_y = values_y[i]
        ibin, jbin = searchsorted_devfunc2D(edges_x, edges_y, v_x, v_y)
        if ibin>=0 and ibin < contents.shape[0] and jbin>=0 and jbin < contents.shape[1]:
            out[i] = contents[ibin, jbin]

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_cudakernel[32, 1024](values, edges, contents, out)

def get_bin_contents2D(values_x, values_y, edges, contents, out):
    get_bin_contents2D_cudakernel(values_x, values_y, edges[0], edges[1], contents, out)

def calc_px(content_pt, content_phi):
    out = cupy.zeros(content_pt.shape[0] - 1, dtype=cupy.float32)
    calc_px_cudakernel(content_pt, content_phi, out)
    cuda.synchronize()
    return out

def calc_py(content_pt, content_phi):
    out = cupy.zeros(content_pt.shape[0] - 1, dtype=cupy.float32)
    calc_py_cudakernel(content_pt, content_phi, out)
    cuda.synchronize()
    return out

def calc_pz(content_pt, content_eta):
    out = cupy.zeros(content_pt.shape[0] - 1, dtype=cupy.float32)
    calc_pz_cudakernel(content_pt, content_eta, out)
    cuda.synchronize()
    return out

def calc_en(content_pt, content_eta, content_mass):
    out = cupy.zeros(content_pt.shape[0] - 1, dtype=cupy.float32)
    calc_en_cudakernel(content_pt, content_eta, content_mass, out)
    cuda.synchronize()
    return out

# functions preparing inputs for COBRA DNN architecture (not nice, but it works!!!)
def make_jets_inputs(content, offsets, nobj, feats, mask_rows, mask_content):

    out = cupy.zeros((len(offsets) - 1, nobj, len(feats)), dtype=cupy.float32)
    for f in feats:
        if f == "px":
            feature = calc_px(content.pt, content.phi)
        elif f == "py":
            feature = calc_py(content.pt, content.phi)
        elif f == "pz":
            feature = calc_pz(content.pt, content.eta)
        elif f == "en":
            feature = calc_en(content.pt, content.eta, content.mass)
        else:
            feature = getattr(content, f)
        dnn_jets_cudakernel(feature, offsets, feats.index(f), nobj, mask_rows, mask_content, out)
    cuda.synchronize()
    return out

def make_leps_inputs(electrons, muons, numEvents, feats, mask_rows, el_mask_content, mu_mask_content):

    inds = cupy.zeros(numEvents, dtype=cupy.int32)

    feature = {}
    feature["pt"] = get_in_offsets(muons.pt, muons.offsets, inds, mask_rows, mu_mask_content) + get_in_offsets(electrons.pt, electrons.offsets, inds, mask_rows, el_mask_content)
    feature["eta"] = get_in_offsets(muons.eta, muons.offsets, inds, mask_rows, mu_mask_content) + get_in_offsets(electrons.eta, electrons.offsets, inds, mask_rows, el_mask_content)
    feature["phi"] = get_in_offsets(muons.phi, muons.offsets, inds, mask_rows, mu_mask_content) + get_in_offsets(electrons.phi, electrons.offsets, inds, mask_rows, el_mask_content)
    feature["mass"] = get_in_offsets(muons.mass, muons.offsets, inds, mask_rows, mu_mask_content) + get_in_offsets(electrons.mass, electrons.offsets, inds, mask_rows, el_mask_content)

    out = cupy.zeros((numEvents, 1, len(feats)), dtype=cupy.float32)
    for f in feats:
        if f == "px":
            feature["px"] = calc_px(feature["pt"], feature["phi"])
        elif f == "py":
            feature["py"] = calc_py(feature["pt"], feature["phi"])
        elif f == "pz":
            feature["pz"] = calc_pz(feature["pt"], feature["eta"])
        elif f == "en":
            feature["en"] = calc_en(feature["pt"], feature["eta"], feature["mass"])
        dnn_leps_cudakernel(feature[f], feats.index(f), mask_rows, out)
    cuda.synchronize()
    return out

def make_met_inputs(content, numEvents, feats, mask_rows):

    out = cupy.zeros((numEvents, len(feats)), dtype=cupy.float32)
    for f in feats:
        if f == "px":
            feature = calc_px(content["MET_pt"], content["MET_phi"])
        elif f == "py":
            feature = calc_py(content["MET_pt"], content["MET_phi"])
        else:
            feature = content["MET_" + f]
        dnn_met_cudakernel(feature, feats.index(f), mask_rows, out)
    cuda.synchronize()
    return out

