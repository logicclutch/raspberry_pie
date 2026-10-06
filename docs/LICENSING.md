# Licences

The software runs only on devices listed in a licence made by LogicClutch. A **licence key** is one line
of text starting with `ANPR1-`. There are two kinds:

- **One key for all machines** (one per customer): lists all of that customer's machine IDs, so the same
  key is pasted on every one of their devices. Simplest.
- **One key per machine**: each device gets its own, shorter key. The vendor downloads them all as one
  Excel file (or CSV) with machine ID, label and key per row.

Either way the device checks that its own machine ID is in the key. The key and a `licence.json` file are
the same licence in two forms. No internet is needed: the device checks the key itself.

- The file is **signed**. Changing anything in it (customer, end date, one more serial) makes it invalid.
- Each device is identified by its **Raspberry Pi serial number** (the "device ID").
- A licence can have an **end date**, or none. Setting the Pi's clock back does not extend it.
- Without a valid licence the engine does nothing: no camera, no plates, nothing sent. It starts within
  seconds of activation, and a renewed key is picked up within seconds too, with no reinstall or restart.
- The dashboard footer shows who the device is licensed to and until when. It turns orange in the last
  30 days and red when there is a problem.

---

## Part 1: for the client (what to do on each Pi)

### On the dashboard (easiest)

Open the device's dashboard (`http://<device-ip>:8000`). A device without a licence opens the
**Activate this device** box by itself (or click the licence text at the bottom of the page).

1. **Copy the machine ID** shown there and send it to LogicClutch, with the IDs of all your other devices.
2. You receive either **one key for all devices**, or an **Excel list with one key per device**. Paste the
   key (for this device: the row with its machine ID) in the box and press **Activate**.
3. The dashboard asks for the **admin password** (the installer prints it; it is `web.admin_token` in
   `/opt/anpr/config.yaml`). People who can only view the dashboard cannot activate.
4. The device checks that its machine ID is in the key, saves it, and starts within a few seconds.

To renew or add devices, paste the new key the same way. The public view-only link cannot activate.

### From the command line (devices without a browser)

**1. Find the device ID** of each Pi and send the list to LogicClutch:

```
cd /opt/anpr && .venv/bin/python -m anpr.engine --machine-id
```

(The same number is the `Serial` line in `cat /proc/cpuinfo`. The installer also prints it at the end.)

**2. Activate** with the key you receive (or `sudo cp licence.json /opt/anpr/licence.json`):

```
cd /opt/anpr && sudo -u anpr .venv/bin/python -m anpr.engine --config /opt/anpr/config.yaml --activate 'ANPR1-...'
```

The engine starts within a few seconds. Nothing else is needed.

**3. Check it** (optional):

```
cd /opt/anpr && .venv/bin/python -m anpr.engine --config /opt/anpr/config.yaml --check-licence
```

| Message | Meaning / what to do |
|---|---|
| `licence OK: ... valid until 2027-09-30` | All good |
| `no licence file at /opt/anpr/licence.json` | Copy the licence file (step 2) |
| `not for this device (device ID ...)` | This Pi is not in the licence. Send its device ID to LogicClutch |
| `ended on ...` | The licence has ended. Ask LogicClutch for a renewed one |
| `signature is not valid` | The file was changed or damaged. Copy the original file again |
| `the clock of this device is wrong` | Set the correct date and time (connect it to the internet) |

Adding Pis later, or renewing, means a new licence file. Copy it over the old one in the same place.

---

## Part 2: for LogicClutch only (issuing licences)

**Never give a client anything from `tools/` or the key file.** The installer does not copy `tools/`.

### Licence Manager (dashboard)

```
cd ~/Desktop/raspberry-main
.venv/bin/python tools/licence_dashboard.py
```

It opens in the browser (the link is also printed; it changes every time). Then:

1. **Customer name**, and paste all their **machine IDs**, one per line. Text after `#` is a label for
   that machine, e.g. `00000000a1b2c3d4  # Gate 1`.
2. **Key type**: **One key for all machines**, or **One key per machine**.
3. **Valid for**: 1 / 2 / 3 years, until a date, or no end date. Optional note (order number).
4. **Generate**:
   - one key for all: **Copy licence key** and send it (or **Download licence.json**);
   - one key per machine: **Download all keys (Excel)** (or **CSV**) and send the file. Each row has the
     machine ID, its label and its key. The Excel file keeps machine IDs exactly as they are.
   Thousands of per-machine keys take a while to make (about 40 s for 5000).

**Issued licences** lists everything you made, with key type and status (Active / Ending soon / Ended).
A per-machine batch is one row. Click a row to see its machines and copy keys again (or download the
Excel file again). **Renew / add machines** starts a new licence with the same
customer and machines: add IDs or pick new dates and generate. **Check a licence key** tells you what
is inside any key a customer sends back, and whether it is valid for a machine ID.

The Licence Manager runs **on this computer only** (127.0.0.1), needs its secret link, and refuses
other websites. Never run it on a server or put it behind Cloudflare.

### Command line (same licences)

### The signing key

- It is at `~/anpr-licensing/vendor_key.json` on the vendor Mac (made once with `keygen`).
- **Back it up** (e.g. an encrypted USB stick in a safe place). If it is lost, no new licences can be
  made for software already delivered. If it leaks, anyone can make licences.
- Its public half is built into the software (`PUBLIC_KEY_HEX` in `anpr/licence.py`). Never run `keygen`
  again: a new key would make every licence already issued stop working.

### Make a licence

Put the customer's device IDs in a text file, one per line (`#` comments and leading zeros are fine):

```
# Acme Logistics, Jaipur warehouse
00000000a1b2c3d4   # gate 1
00000000b5c6d7e8   # gate 2
```

Then:

```
cd ~/Desktop/raspberry-main
.venv/bin/python tools/licence_tool.py issue \
    --customer "Acme Logistics" --devices-file acme_serials.txt --days 365 --out acme_licence.json
```

- Validity: `--days 365` (from today), or `--expires 2027-09-30` (last valid day), or `--no-expiry`.
- A few devices can also be given directly: `--devices 00000000a1b2c3d4,00000000b5c6d7e8`.
- `--note "PO 4512"` is kept inside the licence.
- A copy of every licence issued is kept in `~/anpr-licensing/issued/`.

Send the customer the `.json` file. For more Pis or a renewal, issue a new file with the full list.

### Check a licence file

```
.venv/bin/python tools/licence_tool.py show acme_licence.json --device 00000000a1b2c3d4
```

### Limits (be realistic)

- Someone with the skills and time to change the software itself could remove the check. Compiling
  the software (planned for the installer) makes this much harder. The written licence agreement is
  the legal protection.
- If a Pi's clock was once set far in the future, the licence counts from that date. Fix the clock
  and reset it with: `sqlite3 /opt/anpr/data/anpr.db "DELETE FROM settings WHERE key='licence_clock'"`.
