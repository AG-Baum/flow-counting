mod paper;
pub mod slow_ref;
use hdf5::file::File;
use std::cmp::min;

pub struct Event {
    x: u16,
    y: u16,
    time: i64,
    intens: u16,
}

pub struct Trigger{
    time: i64,
    type_: u8,
}

pub struct Clust {
    x: u16,
    y: u16,
    time: i64,
    size: u16,
    pub sum: u16,
    intens: u16,
}

pub fn write_hdf5(path: &str, clusters: &[Clust], triggers: &[Trigger],
                  frames: &[u64]) {
    let num_events = clusters.len();
    let mut data_x = Vec::<u16>::with_capacity(num_events);
    let mut data_y = Vec::<u16>::with_capacity(num_events);
    let mut data_size = Vec::<u16>::with_capacity(num_events);
    let mut data_time = Vec::<i64>::with_capacity(num_events);
    let mut data_intens = Vec::<u16>::with_capacity(num_events);
    let mut data_sum = Vec::<u16>::with_capacity(num_events);
    for clust in clusters {
        data_x.push(clust.x);
        data_y.push(clust.y);
        data_size.push(clust.size);
        data_time.push(clust.time);
        data_intens.push(clust.intens);
        data_sum.push(clust.sum);
    }

    // Trigger arrays
    let num_triggers = triggers.len();
    let mut data_tdc_time = Vec::<i64>::with_capacity(num_triggers);
    let mut data_tdc_type = Vec::<u8>::with_capacity(num_triggers);
    for trig in triggers {
        data_tdc_time.push(trig.time);
        data_tdc_type.push(trig.type_);
    }

    let file = File::create(path).unwrap(); // open for writing
    // let group = file.create_group("dir").unwrap(); // create a group
    let builder = file.new_dataset_builder();
    builder.with_data(&data_x).create("x").unwrap();
    let builder = file.new_dataset_builder();
    builder.with_data(&data_y).create("y").unwrap();
    let builder = file.new_dataset_builder();
    builder.with_data(&data_size).create("size").unwrap();
    let builder = file.new_dataset_builder();
    builder.with_data(&data_time).create("time").unwrap();
    let builder = file.new_dataset_builder();
    builder
        .with_data(&data_intens)
        .create("max_intens")
        .unwrap();
    let builder = file.new_dataset_builder();
    builder.with_data(&data_sum).create("sum_intens").unwrap();

    let builder = file.new_dataset_builder();
    builder.with_data(&data_tdc_time).create("trigger_time").unwrap();
    let builder = file.new_dataset_builder();
    builder.with_data(&data_tdc_type).create("trigger_type").unwrap();

    let builder = file.new_dataset_builder();
    builder.with_data(&frames).create("frame_time").unwrap();

    file.flush().unwrap();
}

/*
 * Loads Events from a hdf5 file.
 */
pub fn load_hdf5(path: &str) -> Result<(Vec<Event>, Vec<Trigger>, Vec<u64>), Box<dyn std::error::Error>> {
    let file: File = File::open(path)?;
    let ds = file.dataset("/x")?; // open the datasets
    let data_x = ds.read_1d::<u16>()?;

    let ds = file.dataset("/y")?; // open the datasets
    let data_y = ds.read_1d::<u16>()?;

    let ds = file.dataset("/tot")?; // open the datasets
    let data_tot = ds.read_1d::<u16>()?;

    let ds = file.dataset("/toa")?; // open the datasets
    let data_toa = ds.read_1d::<i64>()?;

    let ds = file.dataset("/tdc_time")?; // open the datasets
    let data_tdc_time = ds.read_1d::<i64>()?;

    let ds = file.dataset("/tdc_time")?; // open the datasets
    let data_tdc_type = ds.read_1d::<u8>()?;

    let ds = file.dataset("/frame_time")?;
    let data_frame_time = ds.read_1d::<u64>()?;

    let num_events = min(
        min(data_x.len(), data_y.len()),
        min(data_toa.len(), data_tot.len()),
    );
    let mut event_vec = Vec::<Event>::with_capacity(num_events);

    for index in 0..num_events {
        let x = data_x[index];
        let y = data_y[index];
        let tot = data_tot[index];
        let toa = data_toa[index];
        let st = Event {
            x: x,
            y: y,
            time: toa,
            intens: tot,
        };
        event_vec.push(st);
    }

    // Triggers

    let num_triggers = data_tdc_time.len();
    let mut trig_vec = Vec::<Trigger>::with_capacity(num_triggers);

    for index in 0..num_triggers {
        let time = data_tdc_time[index];
        let type_ = data_tdc_type[index];
        let st = Trigger {
            time: time,
            type_: type_,
        };
        trig_vec.push(st)
    }

    // Frames
    let frame_vec = data_frame_time.to_vec();

    return Ok((event_vec, trig_vec, frame_vec));
}
/*
 * Calculates abs(x-y) of unsigned integer variables
 *
*/
#[inline]
fn abs_diff(x: u16, y: u16) -> u16 {
    return u16::saturating_sub(x, y) + u16::saturating_sub(y, x);
}

/*
 * Cluster analysis using a cutoff and no dynamic buffer size
 */
pub fn clust_analysis_cutoff(
    hits: &[Event],
    eps_space: u16,
    eps_time: f64,
    cut_off: usize,
) -> Vec<Clust> {
    let eps_time_count = (eps_time / 1e-12) as i64; // time is kept as int for faster computation

    let mut extracted_cluster = Vec::<Clust>::with_capacity(hits.len() / 2);

    'outer: for hit in hits {
        let total_len = extracted_cluster.len();
        let start_idx = total_len.saturating_sub(cut_off);
        for idx in start_idx..total_len {
            let clust = &mut extracted_cluster[idx];
            let clust_is_current = (hit.time - clust.time).abs() <= eps_time_count;
            if !clust_is_current {
                continue;
            }

            let close_to_cluster = abs_diff(hit.x, clust.x) + abs_diff(hit.y, clust.y) <= eps_space;
            if close_to_cluster {
                clust.size += 1;
                clust.sum += hit.intens;
                clust.time = min(hit.time, clust.time);
                if clust.intens < hit.intens {
                    clust.intens = hit.intens;
                    clust.x = hit.x;
                    clust.y = hit.y;
                }
                continue 'outer;
            }
        }
        let clust = Clust {
            x: hit.x,
            y: hit.y,
            time: hit.time,
            size: 1,
            sum: hit.intens,
            intens: hit.intens,
        };
        extracted_cluster.push(clust);
    }
    return extracted_cluster;
}
/*
 * Cluster analysis of the hits.
 */
pub fn clust_analysis(hits: &[Event], eps_space: u16, eps_time: f64) -> Vec<Clust> {
    let eps_time_count = (eps_time / 1e-12) as i64; // time is kept as int for faster computation

    let mut extracted_cluster = Vec::<Clust>::with_capacity(hits.len() / 2);

    let mut oldest_index = 0;
    'outer: for hit in hits {
        for idx in oldest_index..extracted_cluster.len() {
            let clust = &mut extracted_cluster[idx];
            let clust_is_current = (hit.time - clust.time).abs() <= eps_time_count;
            if !clust_is_current {
                oldest_index = idx;
                continue;
            }
            let close_to_cluster = abs_diff(hit.x, clust.x) + abs_diff(hit.y, clust.y) <= eps_space;
            if close_to_cluster {
                clust.size += 1;
                clust.sum += hit.intens;
                clust.time = min(hit.time, clust.time);
                if clust.intens < hit.intens {
                    clust.intens = hit.intens;
                    clust.x = hit.x;
                    clust.y = hit.y;
                }
                continue 'outer;
            }
        }
        let clust = Clust {
            x: hit.x,
            y: hit.y,
            time: hit.time,
            size: 1,
            sum: hit.intens,
            intens: hit.intens,
        };
        extracted_cluster.push(clust);
    }
    return extracted_cluster;
}

#[cfg(test)]
mod tests {
    use crate::{
        clust_analysis, clust_analysis_cutoff, load_hdf5, slow_ref::clust_analysis_with_ring,
    };

    #[test]
    fn compare_impl() {
        let path = "./example_measurement.hdf5";
        let hits = load_hdf5(&path).unwrap();
        let clusters = clust_analysis(&hits, 5, 500e-9);
        let clusters_cutoff = clust_analysis_cutoff(&hits, 5, 500e-9, 5);
        let clusters_ring = clust_analysis_with_ring(&hits, 5, 500e-9);
        assert_eq!(clusters_ring.len(), clusters.len());
        for (c1, c2) in clusters.iter().zip(clusters_ring.iter()) {
            assert_eq!(c1.x, c2.x);
            assert_eq!(c1.y, c2.y);
            assert_eq!(c1.intens, c2.intens);
        }
        assert_eq!(clusters_cutoff.len(), clusters.len());
        for (c1, c2) in clusters.iter().zip(clusters_cutoff.iter()) {
            assert_eq!(c1.x, c2.x);
            assert_eq!(c1.y, c2.y);
            assert_eq!(c1.intens, c2.intens);
        }
    }
}
