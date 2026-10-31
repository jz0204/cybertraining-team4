// Exercise the viewer's actual polygon hit-test, including a courtyard hole.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../la_best_preview.html'), 'utf8');
const definition = html.match(/function inside\(x,y,rings\)\{[^\n]+\}/)[0];
const context = {};
vm.createContext(context);
vm.runInContext(definition, context);
const outer = [[0,0],[1,0],[1,1],[0,1],[0,0]];
const hole = [[.3,.3],[.7,.3],[.7,.7],[.3,.7],[.3,.3]];
assert.equal(context.inside(.2,.2,[outer]), true);
assert.equal(context.inside(1.2,.2,[outer]), false);
assert.equal(context.inside(.5,.5,[outer,hole]), false);
assert.equal(context.inside(.2,.2,[outer,hole]), true);
console.log('PASS actual scene hit-test, including interior holes');
