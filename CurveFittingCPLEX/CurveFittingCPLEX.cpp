// CurveFittingCPLEX.cpp: entry point of the application.
//

#include "CurveFittingCPLEX.h"

ILOSTLBEGIN

using namespace std;

// Stores input data points (xi, yi)
class DataSet {
public:
    vector<double> x;
    vector<double> y;

    DataSet() {
        x = {
            0.0,0.5,1.0,1.5,1.9,2.5,3.0,3.5,4.0,4.5,
            5.0,5.5,6.0,6.6,7.0,7.6,8.5,9.0,10.0
        };

        y = {
            1.0,0.9,0.7,1.5,2.0,2.4,3.2,2.0,2.7,3.5,
            1.0,4.0,3.6,2.7,5.7,4.6,6.0,6.8,7.3
        };
    }

    // number of data points
    int size() const {
        return x.size();
    }
};

class CurveFittingSolver {
private:
    IloEnv& env;
    const DataSet& data;

public:
    CurveFittingSolver(IloEnv& env, const DataSet& data)
        : env(env), data(data) {
    }

    // 1. Linear fit y = bx + a, minimize sum |yi - (bxi + a)|
    void solveLinearSumAbs() {
        IloModel model(env);

        // line parameters
        IloNumVar a(env, -IloInfinity, IloInfinity);
        IloNumVar b(env, -IloInfinity, IloInfinity);

        int n = data.size();

        // auxiliary variables for absolute deviations
        IloNumVarArray d(env, n, 0, IloInfinity);

        for (int i = 0; i < n; i++) {
            IloExpr pred = b * data.x[i] + a;

            model.add(data.y[i] - pred <= d[i]);
            model.add(pred - data.y[i] <= d[i]);

            pred.end();
        }

        // objective: minimize sum of deviations
        IloExpr obj(env);
        for (int i = 0; i < n; i++)
            obj += d[i];

        model.add(IloMinimize(env, obj));
        obj.end();

        cout << "\n=====================================\n";
        cout << "Method 1: Linear fit (min sum abs deviations)\n";
        cout << "Model: y = bx + a\n";
        cout << "=====================================\n";

        IloCplex cplex(model);
        if (!cplex.solve()) {
            cout << "No solution found" << endl;
            return;
        }

        cout << "\nLinear |sum deviations|" << endl;
        cout << "a = " << cplex.getValue(a) << endl;
        cout << "b = " << cplex.getValue(b) << endl;
    }

    // 2. Linear fit y = bx + a, minimize max |yi - (bxi + a)|
    void solveLinearMaxDev() {
        IloModel model(env);

        // line parameters
        IloNumVar a(env, -IloInfinity, IloInfinity);
        IloNumVar b(env, -IloInfinity, IloInfinity);

        int n = data.size();

        // variable representing maximum deviation
        IloNumVar t(env, 0, IloInfinity);

        for (int i = 0; i < n; i++) {
            IloExpr pred = b * data.x[i] + a;

            model.add(data.y[i] - pred <= t);
            model.add(pred - data.y[i] <= t);

            pred.end();
        }

        // objective: minimize maximum deviation
        model.add(IloMinimize(env, t));

        cout << "\n=====================================\n";
        cout << "Method 2: Linear fit (min max deviation)\n";
        cout << "Model: y = bx + a\n";
        cout << "=====================================\n";

        IloCplex cplex(model);
        if (!cplex.solve()) {
            cout << "No solution found" << endl;
            return;
        }

        cout << "\nLinear |max deviation|" << endl;
        cout << "a = " << cplex.getValue(a) << endl;
        cout << "b = " << cplex.getValue(b) << endl;
        cout << "max deviation = " << cplex.getValue(t) << endl;
    }

    // 3.1. Quadratic fit y = cx^2 + bx + a, minimize sum |yi - (cxi^2 + bxi + a)|
    void solveQuadraticSumAbs() {
        IloModel model(env);

        // quadratic coefficients
        IloNumVar a(env, -IloInfinity, IloInfinity);
        IloNumVar b(env, -IloInfinity, IloInfinity);
        IloNumVar c(env, -IloInfinity, IloInfinity);

        int n = data.size();

        // absolute deviation variables
        IloNumVarArray d(env, n, 0, IloInfinity);

        for (int i = 0; i < n; i++) {
            IloExpr pred = c * data.x[i] * data.x[i] +
                b * data.x[i] +
                a;

            model.add(data.y[i] - pred <= d[i]);
            model.add(pred - data.y[i] <= d[i]);

            pred.end();
        }

        // objective: minimize total deviation
        IloExpr obj(env);
        for (int i = 0; i < n; i++)
            obj += d[i];

        model.add(IloMinimize(env, obj));
        obj.end();

        cout << "\n=====================================\n";
        cout << "Method 3.1: Quadratic fit (min sum abs deviations)\n";
        cout << "Model: y = cx^2 + bx + a\n";
        cout << "=====================================\n";

        IloCplex cplex(model);
        if (!cplex.solve()) {
            cout << "No solution found" << endl;
            return;
        }

        cout << "\nQuadratic |sum deviations|" << endl;
        cout << "a = " << cplex.getValue(a) << endl;
        cout << "b = " << cplex.getValue(b) << endl;
        cout << "c = " << cplex.getValue(c) << endl;
    }

    // 3.2. Quadratic fit y = cx^2 + bx + a, minimize max |yi - (cxi^2 + bxi + a)|
    void solveQuadraticMaxDev() {
        IloModel model(env);

        // quadratic coefficients
        IloNumVar a(env, -IloInfinity, IloInfinity);
        IloNumVar b(env, -IloInfinity, IloInfinity);
        IloNumVar c(env, -IloInfinity, IloInfinity);

        int n = data.size();

        // variable for maximum deviation
        IloNumVar t(env, 0, IloInfinity);

        for (int i = 0; i < n; i++) {
            IloExpr pred = c * data.x[i] * data.x[i] +
                b * data.x[i] +
                a;

            model.add(data.y[i] - pred <= t);
            model.add(pred - data.y[i] <= t);

            pred.end();
        }

        // objective: minimize maximum deviation
        model.add(IloMinimize(env, t));

        cout << "\n=====================================\n";
        cout << "Method 3.2: Quadratic fit (min max deviation)\n";
        cout << "Model: y = cx^2 + bx + a\n";
        cout << "=====================================\n";

        IloCplex cplex(model);
        if (!cplex.solve()) {
            cout << "No solution found" << endl;
            return;
        }

        cout << "\nQuadratic |max deviation|" << endl;
        cout << "a = " << cplex.getValue(a) << endl;
        cout << "b = " << cplex.getValue(b) << endl;
        cout << "c = " << cplex.getValue(c) << endl;
        cout << "max deviation = " << cplex.getValue(t) << endl;
    }
};

int main() {
    IloEnv env;

    try {
        DataSet data;

        CurveFittingSolver solver(env, data);

        solver.solveLinearSumAbs();
        solver.solveLinearMaxDev();
        solver.solveQuadraticSumAbs();
        solver.solveQuadraticMaxDev();
    }
    catch (IloException& e) {
        cerr << "CPLEX error: " << e << endl;
    }

    env.end();

    return 0;
}