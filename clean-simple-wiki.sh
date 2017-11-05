#!/usr/bin/env bash

sed -i.old -E '/^HEADING|^.{,3}$|^.{700}/d' full-simple-wiki.txt
