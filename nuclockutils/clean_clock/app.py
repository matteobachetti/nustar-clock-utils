import glob
import os
import copy
import time
from functools import lru_cache
import pickle

import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy import log
from nuclockutils.nustarclock import NUSTAR_MJDREF, get_bad_points_db
from nuclockutils.nustarclock import temperature_correction_table, \
    flag_bad_points

from nuclockutils.utils import get_obsid_list_from_heasarc, \
    aggregate_all_tables, merge_and_sort_arrays, eliminate_array_from_array, \
    find_idxs, filter_dict_with_re, cross_two_gtis, sec_to_mjd, sec_to_ut

from nuclockutils.nustarclock import load_temptable, load_freq_changes, \
    load_and_flag_clock_table, find_good_time_intervals, calculate_stats, \
    eliminate_trends_in_residuals

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from textwrap import dedent as d
import json


CLOCKFILE='latest_clock.dat'
TEMPFILE='tcxo_tmp_archive.csv'
FREQFILE='latest_freq.dat'
MODELVERSION=None


def recalc(outfile='save_all.pickle'):
    t0 = time.time()
    temptable_raw = \
        load_temptable(TEMPFILE)

    log.info("Querying history of NuSTAR observations...")
    all_nustar_obs = get_obsid_list_from_heasarc()
    all_nustar_obs.sort('MET')

    all_nustar_obs['text'] = [
        (f"Source: {aobs['NAME']}<br>"+
         f"ObsID: {aobs['OBSID']}<br>"+
         f"Start: {aobs['DATE']} (MJD {aobs['TIME']})<br>"+
         f"End: {aobs['DATE-END']} (MJD {aobs['END_TIME']})<br>"+
         f"Type: {aobs['OBS_TYPE']}<br>"+
         f"Mode: {aobs['OBSERVATION_MODE']}<br>")
        for aobs in all_nustar_obs]

    clock_offset_table = \
        load_and_flag_clock_table(clockfile=CLOCKFILE, shift_non_malindi=False)
    clock_offset_table_corr = \
        load_and_flag_clock_table(clockfile=CLOCKFILE, shift_non_malindi=True)

    table_times = temptable_raw['met']
    met_start = clock_offset_table['met'][0]
    met_stop = clock_offset_table['met'][-1] + 30
    clock_jump_times = \
        np.array([78708320, 79657575, 81043985, 82055671, 293346772,
                  392200784, 394825882, 395304135, 407914525, 408299422])
    clock_jump_times += 30 #  Sum 30 seconds to avoid to exclude these points
                           #  from previous interval

    gtis = find_good_time_intervals(temptable_raw, clock_jump_times)

    table_new = temperature_correction_table(
        met_start, met_stop, temptable=temptable_raw,
        freqchange_file=FREQFILE,
        time_resolution=10, craig_fit=False, hdf_dump_file='dump.hdf5', version=MODELVERSION)

    gtis = cross_two_gtis(gtis, np.asarray([[table_new['met'][0] - 1, table_new['met'][-1] + 1]]))

    table_new = eliminate_trends_in_residuals(
        table_new, clock_offset_table_corr, gtis,
        fixed_control_points=np.arange(291e6, 295e6, 86400))

    mets = np.array(table_new['met'])
    start = mets[0]
    stop = mets[-1]

    good_mets = clock_offset_table['met'] < stop
    clock_offset_table = clock_offset_table[good_mets]

    clock_mets = clock_offset_table['met']
    clock_mjds = clock_offset_table['mjd']
    dates = Time(clock_mjds[:-1], format='mjd')

    tempcorr_idx = \
        np.searchsorted(table_new['met'], clock_offset_table['met'])

    clock_residuals = \
        np.array(
            clock_offset_table['offset'] - table_new['temp_corr'][tempcorr_idx]
        )
    clock_residuals_detrend = np.array(
        clock_offset_table['offset'] - table_new['temp_corr_detrend'][tempcorr_idx])

    all_data = Table({'met': clock_mets[:-1],
                      'mjd': np.array(clock_mjds[:-1], dtype=int),
                      'doy': dates.strftime("%Y:%j"),
                      'utc': dates.strftime("%Y:%m:%d"),
                      'offset': clock_offset_table['offset'][:-1],
                      'residual': clock_residuals[:-1],
                      'residual_detrend': clock_residuals_detrend[:-1],
                      'station': clock_offset_table['station'][:-1]})

    all_data.meta['clock_offset_file'] = CLOCKFILE
    all_data.meta['temperature_file'] = TEMPFILE
    all_data.meta['frequency_file'] = FREQFILE

    pickle.dump((all_data, table_new, gtis, all_nustar_obs),
                open(outfile, 'wb'))

    calculate_stats(all_data)

    log.info(f"Reprocessing done. It took {time.time() - t0} s")
    return all_data, table_new, gtis, all_nustar_obs


def plot_dash(all_data, table_new, gti, all_nustar_obs,
    axis_ranges=None, met_range=None):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    log.info("Updating plot")

    temptable_data = load_temptable(all_data.meta['temperature_file'])

    if met_range is not None:
        met_min, met_max = met_range
        temptable_data = \
            temptable_data[find_idxs(temptable_data['met'], met_min, met_max)]
        table_new = \
            table_new[find_idxs(table_new['met'], met_min, met_max)]

    all_data = flag_bad_points(all_data)

    temptable_data, table_new = \
        aggregate_all_tables([temptable_data, table_new], max_number=300)

    bad = all_data['flag'] == True

    hovertemplate=('MET: %{x:d}<br>'
                   '%{text}')

    # Add a final line to include overflow point
    all_nustar_obs.add_row(all_nustar_obs[-1])
    for col in all_nustar_obs.colnames:
        if isinstance(all_nustar_obs[col][0], str):
            all_nustar_obs[col][-1] = ""
        else:
            all_nustar_obs[col][-1] *= 0

    idx = np.searchsorted(all_nustar_obs['MET'][1:], table_new['met'])

    all_nustar_obs_reindex = all_nustar_obs[idx]
    text = np.array([f"MJD {sec_to_mjd(met)}<br>UT {sec_to_ut(met)}<br><br>{string}" for met, string in zip(table_new["met"], all_nustar_obs_reindex['text'])])

    fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.append_trace(go.Scattergl({
        'x': table_new['met'],
        'y': table_new['temp_corr_raw'] * 1e3,
        'hovertemplate': hovertemplate,
        'text': text,
        'mode': 'lines',
        'name': f'Temperature correction raw',
        'marker': {'color': 'grey', 'opacity': 0.5}
    }), 1, 1)
    fig.append_trace(go.Scattergl({
        'x': table_new['met'],
        'y': table_new['temp_corr'] * 1e3,
        'hovertemplate': hovertemplate,
        'text': text,
        'mode': 'lines',
        'name': f'Temperature correction',
        'marker': {'color': 'black'}
    }), 1, 1)
    fig.append_trace(go.Scattergl({
        'x': table_new['met'],
        'y': table_new['temp_corr_trend'] * 1e3,
        'hovertemplate': hovertemplate,
        'text': text,
        'mode': 'lines',
        'showlegend': False,
        'marker': {'color':'black'}
    }), 2, 1)
    for sign in [-1, 1]:
        fig.append_trace(go.Scattergl({
            'x': table_new['met'],
            'y': sign * table_new['std'] * 1e3,
            'hovertemplate': hovertemplate,
            'text': text,
            'mode': 'lines',
            'showlegend': False,
            'marker': {'color':'black'}
         }), 3, 1)

    all_nustar_obs = all_nustar_obs[:-1]

    idx = np.searchsorted(all_nustar_obs['MET'][:-1],
                          np.array(all_data['met']))

    all_nustar_obs_reindex = all_nustar_obs[idx]

    text = np.array([f"MJD {sec_to_mjd(met)}<br>UT {sec_to_ut(met)}<br><br>{string}" for met, string in zip(all_data["met"], all_nustar_obs_reindex['text'])])

    fig.append_trace(go.Scattergl({
        'x': all_data['met'][bad],
        'y': all_data['offset'][bad] * 1e3,
        'hovertemplate': hovertemplate,
        'text': text[bad],
        'mode': 'markers',
        'name': f'Bad clock offset measurements',
        'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
    }), 1, 1)

    all_data_bad = all_data[bad]

    for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
        fig.append_trace(go.Scattergl({
            'x': all_data_bad['met'],
            'y': all_data_bad[ydata] * 1e3,
            'hovertemplate': hovertemplate,
            'text': text[bad],
            'mode': 'markers',
            'showlegend': False,
            'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
         }), row, 1)

    for station, color in zip(['MLD', 'SNG', 'UHI'], ['blue', 'red', 'orange']):
        good = (all_data['station'] == station) & ~bad
        all_data_good = all_data[good]
        fig.append_trace(go.Scattergl({
            'x': all_data_good['met'],
            'y': all_data_good['offset'] * 1e3,
            'hovertemplate': hovertemplate,
            'text': text[good],
            'mode': 'markers',
            'marker': {'color': color, 'size': 3},
            'name': f'Clock offset - {station}'
        }), 1, 1)
        for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
            fig.append_trace(go.Scattergl({
                'x': all_data_good['met'],
                'y': all_data_good[ydata] * 1e3,
                'hovertemplate': hovertemplate,
                'text': text[good],
                'showlegend': False,
                'mode': 'markers',
                'marker': {'color': color, 'size': 3}
            }), row, 1)

    bad_intervals = np.array(
        [[g0, g1] for g0, g1 in zip(gti[:-1, 1], gti[1:, 0])])

    shapes = []
    for bti in bad_intervals:
        if bti[-1] < all_data['met'][0]:
            continue
        if bti[0] > all_data['met'][-1]:
            continue
        shapes.append(dict(type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=max(bti[0], all_data['met'][0]),
                y0=0,
                x1=min(bti[1], all_data['met'][-1]),
                y1=1,
                fillcolor="LightSalmon",
                opacity=0.5,
                layer="below",
                line_width=0,
            ))

    if axis_ranges is not None:
        if 'xaxis.range[0]' in axis_ranges:
            xranges = axis_ranges[f'xaxis.range[0]'], axis_ranges[f'xaxis.range[1]']
            fig.update_xaxes(range=xranges, row=3, col=1)
        for i, row in enumerate([1, 2, 3]):
            label = 'yaxis.range' if row == 1 else f'yaxis{row}.range'
            if label + '[0]' in axis_ranges:
                ranges = axis_ranges[f'{label}[0]'], axis_ranges[f'{label}[1]']
                fig.update_yaxes(row=row, range=ranges, col=1)

    fig.update_xaxes(
        title_text=f"Mission Epoch Time (s from MJD {NUSTAR_MJDREF})",
        row=3, col=1)

    for axis_no, title in zip(
        [1, 2, 3],
        ['Clock offset (ms)', 'Residuals (ms)', 'Detrended Residuals (ms)']):
        # Update yaxis properties
        fig.update_yaxes(title_text=title, row=axis_no, col=1)
        # Add shape regions
        for s in shapes:
            s['xref'] = f'x{axis_no}'
        fig.update_layout(shapes=shapes)

    fig['layout'].update(height=900, legend=dict(orientation="h", x=0, y=1),
                         margin={'t': 20})
    fig['layout']['clickmode'] = 'event+select'

    return fig


def default_axes():
    return {'yaxis.range[0]': -500, 'yaxis.range[1]': 1000,
        'yaxis2.range[0]': -60, 'yaxis2.range[1]': 30,
        'yaxis3.range[0]': -2, 'yaxis3.range[1]': 1}


CURRENT_AXES = default_axes()

def create_app():

    app = dash.Dash(
        __name__,
        assets_folder = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'assets'),
        meta_tags=[{"name": "viewport", "content": "width=device-width"}])


    @lru_cache(maxsize=64)
    def stored_analysis(file):
        if not os.path.exists(file):
            recalc(file)

        log.info(f"Reading data from {file}")
        return pickle.load(open(file, 'rb'))


    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    app.layout = html.Div(className="row", children=[
        html.Div(
            className="three columns div-user-controls",
            children=[
                html.H1(
                    children='NuSTAR Clock offset cleaner',
                    # style={
                    # 'textAlign': 'center',
                    # 'color': colors['text']
                    #     }
                        ),
                dcc.Markdown(d(
                    """
                    To navigate the plot, use the Zoom and Pan tools in the graph's
                    menu bar.
                    [Detailed instructions here](https://plot.ly/chart-studio-help/zoom-pan-hover-controls/)

                    Choose the lasso or rectangle tool and then select points in any
                    of the three plots to mark them as "bad".

                    Select in an empty area to eliminate the current selection.

                    To save the current selection in the bad clock offset database,
                    press "Add to blacklist"

                    To remove the currently selected point from the bad clock offset database,
                    press "Remove from blacklist"

                    When you are done with the selection, press "Recalculate"

                    To undo the current selection, press "Reset"

                    Light salmon areas indicate known issues like major clock readjustments
                    or times where no temperature measurements are present.
                """)),
                html.Div(
                    className="row",
                    children=[
                        html.Button("Reset", id='refresh-button', n_clicks=0),
                        html.Button("Recalculate", id='recalculate-button', n_clicks=0),
                        html.Button("Add to blacklist", id='bad-data-button', n_clicks=0),
                        html.Button("Remove from blacklist", id='actually-good-data-button', n_clicks=0),
                    ]),
                html.Div([
                    html.H2(
                        children='Selected Data',
                            ),
                    html.P(
                        html.Pre(id='selected-data'),)
                ])
            ],

        ),
        html.Div(
            id='figure-div',
            className="six columns div-for-charts bg-grey",
            children=[
                dcc.Graph(figure=plot_dash(*stored_analysis('save_all.pickle'),
                    axis_ranges=CURRENT_AXES), id='my-figure',
                    hoverData={'points': [{'x': 2e8}]}),
            ],
        ),
        html.Div(
            className="three columns div-for-charts bg-grey",
            children=[
            dcc.Graph(id='temperature-time-series'),
            dcc.Graph(id='temperature-gradient-time-series'),
            ]
        ),
        html.Div(id='dummy', style={'display': 'none'}),
        html.Div(id='clock-intermediate-value', style={'display': 'none'}),
        html.Div(id='freq-intermediate-value', style={'display': 'none'}),
        html.Div(id='axis-properties', style={'display': 'none'})
    ])


    @app.callback(
        Output('axis-properties', 'children'),
        [Input('my-figure', 'relayoutData')])
    def update_x_ranges(relayoutData):
        global CURRENT_AXES
        if relayoutData is None:
            raise PreventUpdate

        log.info("Received relayoutData: ", relayoutData)
        rangevals_re = r'^[xy].*\.range.*$'
        xzoom_dict = filter_dict_with_re(relayoutData, rangevals_re)

        rangevals_re = r'^[xy].*\.autorange.*$'
        autorange_dict = filter_dict_with_re(relayoutData, rangevals_re)

        if autorange_dict == {} and xzoom_dict == {}:
            log.info("No range information.")
            raise PreventUpdate

        if autorange_dict != {}:
            CURRENT_AXES = default_axes()
        else:
            new_ranges = xzoom_dict
            for key, value in new_ranges.items():
                CURRENT_AXES[key] = value
        return json.dumps(CURRENT_AXES)


    @app.callback(
        Output('my-figure', 'figure'),
        [Input('axis-properties', 'children')])
    def display_relayout_data(children):
        if children is None:
            raise PreventUpdate

        relayoutData = json.loads(children)

        log.info(f"New relayoutData received: {relayoutData}")

        if relayoutData is None:
            raise PreventUpdate

        all_data, table_new_complete, gti, all_nustar_obs = \
            stored_analysis('save_all.pickle')

        met_range = None
        if 'xaxis.range[0]' in relayoutData:
            met_range = [relayoutData['xaxis.range[0]'],
                         relayoutData['xaxis.range[1]']]

        return plot_dash(
            all_data, table_new_complete, gti, all_nustar_obs,
            axis_ranges=relayoutData, met_range=met_range)


    @app.callback(
        Output("figure-div", "children"),
        [Input("refresh-button", "n_clicks"),
         Input("bad-data-button", "n_clicks"),
         Input("actually-good-data-button", "n_clicks"),
         Input("recalculate-button", "n_clicks"),
         Input('selected-data', 'children')])
    def refresh_output(n_cl_refresh, n_cl_bad, n_cl_good, n_cl_recalc, selectedData):
        global CURRENT_AXES
        ctx = dash.callback_context

        if not ctx.triggered:
            raise PreventUpdate

        who_triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        log.info(f"Pressed {who_triggered}")

        if who_triggered == 'selected-data':
            print("Preventing update")
            raise PreventUpdate

        ALL_BAD_POINTS = get_bad_points_db()
        NEW_BAD_POINTS = []

        if selectedData is not None and "No data selected" not in selectedData:
            selectedData = json.loads(selectedData)
        else:
            selectedData = {}

        if 'points' in selectedData:
            NEW_BAD_POINTS = np.array([p['x'] for p in selectedData['points']])

        if who_triggered == 'recalculate-button':
            log.info("Recalculating all")
            stored_analysis.cache_clear()
            # cache.delete_memoized(stored_analysis, 'save_all.pickle')
            if os.path.exists('save_all.pickle'):
                os.unlink('save_all.pickle')
        elif who_triggered == 'actually-good-data-button' and len(NEW_BAD_POINTS) > 0:
            log.info("Removing point(s) from bad clock offset database")
            ALL_BAD_POINTS = eliminate_array_from_array(
                ALL_BAD_POINTS, NEW_BAD_POINTS)
            np.savetxt('BAD_POINTS_DB.dat', ALL_BAD_POINTS, fmt='%d')
        elif who_triggered == 'bad-data-button' and len(NEW_BAD_POINTS) > 0:
            log.info("Adding point(s) to bad clock offset database")
            ALL_BAD_POINTS = merge_and_sort_arrays(
                ALL_BAD_POINTS, NEW_BAD_POINTS)
            np.savetxt('BAD_POINTS_DB.dat', ALL_BAD_POINTS, fmt='%d')
        else:
            log.info("Refreshing plot")
            CURRENT_AXES = default_axes()

        return [
            dcc.Graph(figure=plot_dash(*stored_analysis('save_all.pickle'),
                axis_ranges=CURRENT_AXES), id='my-figure',
                hoverData={'points': [{'x': 2e8}]}),]


    @app.callback(
        Output('selected-data', 'children'),
        [Input('my-figure', 'selectedData')])
    def display_selected_data(selectedData):
        if selectedData is None:
            return "No data selected"
        else:
            return json.dumps(selectedData, indent=2)


    def create_temperature_timeseries(x, y, axis_type='linear'):
        return {
            'data': [dict(
                x=x,
                y=y,
                mode='lines'
            )],
            'layout': {
                'height': 300,
                'yaxis': {'title': 'TCXO Temperature',
                    'type': 'linear' if axis_type == 'Linear' else 'log'},
                'xaxis': {'title': 'MET', 'showgrid': False,
                'margin':{'t': 20}}

            }
        }


    def create_temperature_gradient_timeseries(x, y, axis_type='linear'):
        return {
            'data': [dict(
                x=x,
                y=y,
                mode='lines'
            )],
            'layout': {
                'height': 300,
                'yaxis': {'title': 'TCXO Temp Gradient',
                    'type': 'linear'},
                'xaxis': {'title': 'MET', 'showgrid': False,
                'margin':{'t': 20}}

            }
        }


    @app.callback(
        [Output('temperature-time-series', 'figure'),
         Output('temperature-gradient-time-series', 'figure')],
        [Input('my-figure', 'hoverData'),
         Input("clock-intermediate-value", "children")])
    def update_temperature_timeseries(hoverData, clockfile):
        if hoverData is None:
            raise PreventUpdate

        temptable = load_temptable(TEMPFILE)
        xstart = hoverData['points'][0]['x'] - 100000
        xstop = hoverData['points'][0]['x'] + 100000

        istart, istop = np.searchsorted(temptable['met'], [xstart, xstop])

        return (create_temperature_timeseries(
            temptable['met'][istart:istop:5],
            temptable['temperature'][istart:istop:5]),
                create_temperature_gradient_timeseries(
            temptable['met'][istart:istop:5],
            temptable['temperature_smooth_gradient'][istart:istop:5]))
    return app


def main(args=None):
    global TEMPFILE
    global CLOCKFILE
    global FREQFILE
    global MODELVERSION
    import argparse
    description = ('Clean clock offset measurements with an handy web '
                   'interface.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--temperature-file", help="Temperature history file",
                        default=None, type=str)
    parser.add_argument("--clock-offset-file", default=None,
                        help="Clock offset history", type=str)
    parser.add_argument("--frequency-file", default=None,
                        help="Divisor frequency history", type=str)
    parser.add_argument("-t", "--tempfile", default=None,
                        help="Temperature file (e.g. the nu<OBSID>_eng.hk.gz "
                             "file in the auxil/directory "
                             "or the tp_tcxo*.csv file)")
    parser.add_argument("--temperature-model-version", default=None, help="Temperature model version")
    args = parser.parse_args(args)
    if args.temperature_file is not None:
        TEMPFILE = args.temperature_file
    if args.clock_offset_file is not None:
        CLOCKFILE = args.clock_offset_file
    if args.frequency_file:
        FREQFILE = args.frequency_file
    if args.temperature_model_version is not None:
        MODELVERSION = args.temperature_model_version

    print("Creating app")
    app = create_app()
    app.run_server(debug=True)


if __name__ == '__main__':
    curdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(curdir, '..', 'tests', 'data')
    CLOCKFILE = os.path.join(datadir, 'sample_clock.dat')
    FREQFILE = os.path.join(datadir, 'sample_freq.dat')
    TEMPFILE = os.path.join(datadir, 'tcxo_tmp_sample.csv')

    main()