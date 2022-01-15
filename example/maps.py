import pandas as pd
import numpy as np

def lookup_cui(medid, meddra_map):
    match = (meddra_map['code'] == int(medid))
    CUI = meddra_map['CUI'][match]
    if CUI.empty:
        print('No CUI code match found for %s, trying with string format: ' % medid)
        match = (meddra_map['code'] == str(medid))
        CUI = meddra_map['CUI'][match]
        if CUI.empty:
            print('No match found for ', medid)
        else:
            return CUI.values[0]
    else:
        return CUI.values[0]


def lookup_ttype(medid, meddra_map):
    match = (meddra_map['code'] == int(medid))
    ttype = meddra_map['term_type'][match]
    if ttype.empty:
        print('No semantic group match found for %s, trying with string format: ' % medid)
        match = (meddra_map['code'] == str(medid))
        ttype = meddra_map['term_type'][match]
        if ttype.empty:
            print('No group match fournd for ', medid)
        else:
            return ttype.values[0]
    else:
        return ttype.values[0]

def match_llt_pt(medid, llt_to_pt):
    match = (llt_to_pt['llt'] == int(medid))
    pt = llt_to_pt['pt'][match]
    if pt.empty:
        print('No preferred term match found for %s, trying with string format: ' % medid)
        match = (llt_to_pt['llt'] == str(medid))
        pt = llt_to_pt['pt'][match]
        if pt.empty:
            print('No preferred term match fournd for ', medid)
        else:
            return pt.values[0]
    else:
        return pt.values[0]

def get_pt_text(medid, meddra_hier):
    match = (meddra_hier['pt'] == int(medid))
    pt_text = meddra_hier['pt_text'][match]
    if pt_text.empty:
        print('No preferred term match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['pt'] == str(medid))
        pt_text = meddra_hier['pt_text'][match]
        if pt_text.empty:
            print('No preferred term match fournd for ', medid)
        else:
            return pt_text.values[0]
    else:
        return pt_text.values[0]

def match_pt_hlt(medid, meddra_hier):
    match = (meddra_hier['pt'] == int(medid))
    hlt = meddra_hier['hlt'][match]
    if hlt.empty:
        print('No high level term match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['pt'] == str(medid))
        hlt = meddra_hier['hlt'][match]
        if hlt.empty:
            print('No high level term match fournd for ', medid)
        else:
            return hlt.values[0]
    else:
        return hlt.values[0]

def get_hlt_text(medid, meddra_hier):
    match = (meddra_hier['hlt'] == int(medid))
    hlt_text = meddra_hier['hlt_text'][match]
    if hlt_text.empty:
        print('No high level term match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['hlt'] == str(medid))
        hlt_text = meddra_hier['hlt_text'][match]
        if hlt_text.empty:
            print('No high level term match fournd for ', medid)
        else:
            return hlt_text.values[0]
    else:
        return hlt_text.values[0]

def match_hlt_hlgt(medid, meddra_hier):
    match = (meddra_hier['hlt'] == int(medid))
    hlgt = meddra_hier['hlgt'][match]
    if hlgt.empty:
        print('No high level group term match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['hlt'] == str(medid))
        hlgt = meddra_hier['hlgt'][match]
        if hlgt.empty:
            print('No high level group term match fournd for ', medid)
        else:
            return hlgt.values[0]
    else:
        return hlgt.values[0]


def get_hlgt_text(medid, meddra_hier):
    match = (meddra_hier['hlgt'] == int(medid))
    hlgt_text = meddra_hier['hlgt_text'][match]
    if hlgt_text.empty:
        print('No high level group term match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['hlgt'] == str(medid))
        hlgt_text = meddra_hier['hlgt_text'][match]
        if hlgt_text.empty:
            print('No high level group term match fournd for ', medid)
        else:
            return hlgt_text.values[0]
    else:
        return hlgt_text.values[0]

def match_hlgt_soc(medid, meddra_hier):
    match = (meddra_hier['hlgt'] == int(medid))
    soc = meddra_hier['soc'][match]
    if soc.empty:
        print('No system organ class match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['hlgt'] == str(medid))
        soc = meddra_hier['soc'][match]
        if soc.empty:
            print('No system organ class  match fournd for ', medid)
        else:
            return soc.values[0]
    else:
        return soc.values[0]

def get_soc_text(medid, meddra_hier):
    match = (meddra_hier['soc'] == int(medid))
    soc_text = meddra_hier['soc_text'][match]
    if soc_text.empty:
        print('No system organ class match found for %s, trying with string format: ' % medid)
        match = (meddra_hier['soc'] == str(medid))
        soc_text = meddra_hier['soc_text'][match]
        if soc_text.empty:
            print('No system organ class match fournd for ', medid)
        else:
            return soc_text.values[0]
    else:
        return soc_text.values[0]

