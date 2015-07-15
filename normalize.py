#!/usr/bin/env python
# -*- coding: utf-8 -*-


def normalize(value, maxi, mini):
    return (value - mini) / (maxi - mini)


def inv_normalize(value, maxi, mini):
    return value * (maxi - mini) + mini
