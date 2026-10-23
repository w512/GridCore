package amdgpu

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// pciIDsFiles are where distributions install the PCI ID database
// (hwdata, pciutils). Tests point it elsewhere.
var pciIDsFiles = []string{
	"/usr/share/hwdata/pci.ids",
	"/usr/share/misc/pci.ids",
	"/usr/share/pci.ids",
}

// deviceName is the board's product name when its FRU EEPROM has one
// (product_name, mostly Pro and Instinct boards), else the PCI ID
// database's name for the chip ("AMD Radeon Pro W7800" from "Navi 31
// [Radeon Pro W7800]"), else the IDs. The name is part of the keys of
// learned model profiles, so it must not depend on the slot or the index.
func deviceName(dir string) string {
	if s, err := readString(dir, "product_name"); err == nil && s != "" {
		return s
	}
	vendor, _ := readString(dir, "vendor")
	device, _ := readString(dir, "device")
	vendor = strings.TrimPrefix(strings.ToLower(vendor), "0x")
	device = strings.TrimPrefix(strings.ToLower(device), "0x")
	for _, path := range pciIDsFiles {
		if s, ok := lookupPCIName(path, vendor, device); ok {
			if open := strings.IndexByte(s, '['); open >= 0 && strings.HasSuffix(s, "]") {
				s = s[open+1 : len(s)-1]
			}
			return "AMD " + s
		}
	}
	return fmt.Sprintf("AMD GPU %s:%s", vendor, device)
}

// lookupPCIName finds the device line of vendor/device in a pci.ids file:
// vendors at column 0, their devices indented by one tab.
func lookupPCIName(path, vendor, device string) (string, bool) {
	if vendor == "" || device == "" {
		return "", false
	}
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer f.Close()
	inVendor := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "" || line[0] == '#':
		case line[0] != '\t':
			if inVendor {
				return "", false
			}
			inVendor = strings.HasPrefix(line, vendor+"  ")
		case inVendor && !strings.HasPrefix(line, "\t\t"):
			if id, name, ok := strings.Cut(line[1:], "  "); ok && strings.EqualFold(id, device) {
				return strings.TrimSpace(name), true
			}
		}
	}
	return "", false
}
